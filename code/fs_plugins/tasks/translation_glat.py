# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass, field
import torch
from fairseq import utils
from fairseq.data import LanguagePairDataset
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset, TranslationConfig
from fairseq.utils import new_arange
from fairseq.dataclass import ChoiceEnum, FairseqDataclass

NOISE_CHOICES = ChoiceEnum(["random_delete", "random_mask", "no_noise", "full_mask"])
GLAT_CHOICES = ChoiceEnum(["no", "glat"])

EVAL_BLEU_ORDER = 4

@dataclass
class TranslationGLATConfig(TranslationConfig):
    noise: NOISE_CHOICES = field(
        default="random_delete",
        metadata={"help": "type of noise"},
    )
    src_upsample_for_task: int = field(
        default=4, metadata={"help": "why CFG? why??? Isn't research hard enough?"}
    )
    glat_mode: GLAT_CHOICES = field(
        default="glat",
        metadata={"help": "with or without glat choice: glat/no"},
    )
    glat_a: float = field(
        default=0.5,
        metadata={"help": "glat a"},
    )
    glat_b: float = field(
        default=0.2,
        metadata={"help": "glat b"},
    )
    max_update: int = field(
        default=300000,
        metadata={"help": "max update"},
    )

@register_task("translation_glat", dataclass=TranslationGLATConfig)
class TranslationGlatTask(TranslationTask):
    cfg: TranslationGLATConfig

    """
    Translation (Sequence Generation) task for Levenshtein Transformer
    See `"Levenshtein Transformer" <https://arxiv.org/abs/1905.11006>`_.
    """

    # @staticmethod
    # def add_cfg(parser):
    #     """Add task-specific arguments to the parser."""
    #     # fmt: off
    #     TranslationTask.add_cfg(parser)
    #     parser.add_argument(
    #         '--glat-mode',
    #         default='no',
    #         choices=['no', 'glat'])
    #     parser.add_argument(
    #         '--glat-a',
    #         default=0.5,
    #         type=float)
    #     parser.add_argument(
    #         '--glat-b',
    #         default=0.2,
    #         type=float)
    #     # fmt: on

    def load_dataset(self, split, epoch=1, combine=False, **kwcfg):
        """Load a given dataset split.

        cfg:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            prepend_bos=True,
        )

    def inject_noise(self, target_tokens):
        def _random_delete(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()

            max_len = target_tokens.size(1)
            target_mask = target_tokens.eq(pad)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(
                target_tokens.eq(bos) | target_tokens.eq(eos), 0.0
            )
            target_score.masked_fill_(target_mask, 1)
            target_score, target_rank = target_score.sort(1)
            target_length = target_mask.size(1) - target_mask.float().sum(
                1, keepdim=True
            )

            # do not delete <bos> and <eos> (we assign 0 score for them)
            target_cutoff = (
                2
                + (
                    (target_length - 2)
                    * target_score.new_zeros(target_score.size(0), 1).uniform_()
                ).long()
            )
            target_cutoff = target_score.sort(1)[1] >= target_cutoff

            prev_target_tokens = (
                target_tokens.gather(1, target_rank)
                .masked_fill_(target_cutoff, pad)
                .gather(1, target_rank.masked_fill_(target_cutoff, max_len).sort(1)[1])
            )
            prev_target_tokens = prev_target_tokens[
                :, : prev_target_tokens.ne(pad).sum(1).max()
            ]

            return prev_target_tokens

        def _random_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_masks = (
                target_tokens.ne(pad) & target_tokens.ne(bos) & target_tokens.ne(eos)
            )
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), unk
            )
            return prev_target_tokens

        def _full_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_mask = (
                target_tokens.eq(bos) | target_tokens.eq(eos) | target_tokens.eq(pad)
            )
            return target_tokens.masked_fill(~target_mask, unk)

        if self.cfg.noise == "random_delete":
            return _random_delete(target_tokens)
        elif self.cfg.noise == "random_mask":
            return _random_mask(target_tokens)
        elif self.cfg.noise == "full_mask":
            return _full_mask(target_tokens)
        elif self.cfg.noise == "no_noise":
            return target_tokens
        else:
            raise NotImplementedError

    def build_generator(self, models, cfg, **unused):
        # add models input to match the API for SequenceGenerator
        from fairseq.iterative_refinement_generator import IterativeRefinementGenerator

        return IterativeRefinementGenerator(
            self.target_dictionary,
            eos_penalty=getattr(cfg, "iter_decode_eos_penalty", 0.0),
            max_iter=getattr(cfg, "iter_decode_max_iter", 0),  # NOTE;
            beam_size=getattr(cfg, "iter_decode_with_beam", 1),
            reranking=getattr(cfg, "iter_decode_with_external_reranker", False),
            decoding_format=getattr(cfg, "decoding_format", None),
            adaptive=not getattr(cfg, "iter_decode_force_max_iter", False),
            retain_history=getattr(cfg, "retain_iter_history", False),
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if constraints is not None:
            # Though see Susanto et al. (ACL 2020): https://www.aclweb.org/anthology/2020.acl-main.325/
            raise NotImplementedError(
                "Constrained decoding with the translation_lev task is not supported"
            )

        return LanguagePairDataset(
            src_tokens, src_lengths, self.source_dictionary, append_bos=True
        )

    def analysis_step(self, model, sample):
        return model.analysis(sample)
    
    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        # print(update_num)
        # -> fuck fairseq! how can I get max_update_num here?
        train_ratio = max(0, min(1, update_num / self.cfg.max_update))
        sample["train_ratio"] = train_ratio
        if self.cfg.glat_mode == "glat":
            sample["glat"] = {"context_p": self.cfg.glat_a - self.cfg.glat_b * train_ratio}
        elif self.cfg.glat_mode == "nlog":
            sample["glat"] = {"schedule": - np.log(train_ratio + 1e-3) * 0.05 + 0.12}
        elif self.cfg.glat_mode == "pcs0":
            if train_ratio < 0.2:
                schedule = 0.5 - 1.5 * train_ratio
            else:
                schedule = - 0.2 / 0.8 * train_ratio + 0.25
            sample["glat"] = {"schedule": schedule}
        elif self.cfg.glat_mode == "pcs1":
            if train_ratio < 0.2:
                schedule = 0.5 - 1.5 * train_ratio
            else:
                schedule = - 0.1 / 0.8 * train_ratio + 0.225
            sample["glat"] = {"schedule": schedule}

        sample["prev_target"] = self.inject_noise(sample["target"])
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            sample["prev_target"] = self.inject_noise(sample["target"])
            loss, sample_size, logging_output = criterion(model, sample)
        if self.cfg.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output
