"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Based on basic.py and https://github.com/tomlimi/MYTE.
"""

import json
from typing import Union
from .base import Tokenizer, get_stats, merge
import unicodedata
from collections import defaultdict


import json
from collections import defaultdict
from typing import Union

class ByteRewriter:
    LEAF = '[LEAF]'

    def __init__(self, rewriting_rules: Union[str, dict[str, str]]):
        if isinstance(rewriting_rules, str):
            with open(rewriting_rules, "r") as f:
                rewriting_rules = json.load(f)
        elif not isinstance(rewriting_rules, dict):
            raise ValueError(f"rewriting_rules should be either a path to json file or a dict, got {type(rewriting_rules)}")

        self.hash_tree = self.construct_hash_tree(rewriting_rules)
        reverse_rewriting_rules = {v: k for k, v in rewriting_rules.items()}
        self.reverse_hash_tree = self.construct_hash_tree(reverse_rewriting_rules)

    def add_leaf(self, hash_tree, byte_in_sequence, byte_out_sequence):
        # Convert hex string sequences to integer lists.
        # Needed because the decompose and merge maps are stored as hex strings
        # but here we work with integer lists.
        byte_in_list = [int(b, 16) for b in byte_in_sequence.split(' ')]
        byte_out_list = [int(b, 16) for b in byte_out_sequence.split(' ')]

        tree_pointer = hash_tree
        for b in byte_in_list:
            if b not in tree_pointer:
                tree_pointer[b] = {}
            tree_pointer = tree_pointer[b]

        tree_pointer[self.LEAF] = byte_out_list

    def construct_hash_tree(self, rewriting_rules):
        hash_tree = defaultdict(dict)
        for b in range(256):  # Initialize with identity rules for single-byte integers
            hash_tree[b][self.LEAF] = [b]

        for in_sequence, out_sequence in rewriting_rules.items():
            self.add_leaf(hash_tree, in_sequence, out_sequence)

        return hash_tree

    def search_hash_tree(self, byte_sequence):
        tree_pointer = self.hash_tree
        for b in byte_sequence:
            if b in tree_pointer:
                tree_pointer = tree_pointer[b]
            else:
                return None

        return tree_pointer.get(self.LEAF)

    def rewrite_bytes(self, in_bytes, reverse=False):
        out_bytes = []
        b_start = 0
        b_end = 0

        while b_start < len(in_bytes):
            tree_pointer = self.hash_tree if not reverse else self.reverse_hash_tree
            for j in range(b_start, len(in_bytes)):
                b = in_bytes[j]
                if b in tree_pointer:
                    tree_pointer = tree_pointer[b]
                elif j == b_start:
                    cur_leaf = [b]
                    b_end = j
                    break
                else:
                    break
                if self.LEAF in tree_pointer:
                    cur_leaf = tree_pointer[self.LEAF]
                    b_end = j
            out_bytes.extend(cur_leaf)
            b_start = b_end + 1

        return out_bytes

    def __repr__(self) -> str:
        return f"ByteRewriter({self.hash_tree})"


class MYTEEncoder:
    def __init__(self, decompose_map='byte_maps/decompose_map.json', merge_map='byte_maps/merge_map.json'):
        self.decompose_rewriter = ByteRewriter(decompose_map)
        self.merge_rewriter = ByteRewriter(merge_map)

    def encode(self, text: str) -> list[int]:
        text_bytes = text.encode("utf-8")
        tokens = list(text_bytes)  # List of integers in range 0..255

        tokens = self.decompose_rewriter.rewrite_bytes(tokens, reverse=False)
        tokens = self.merge_rewriter.rewrite_bytes(tokens, reverse=False)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        if not tokens:  # Check for empty token list
            return ""  # Return empty string if no tokens to decode

        out_tokens = []
        for token in tokens:
            out_tokens.append(token)

        out_tokens = self._morphological_decode(out_tokens)
        return bytes(out_tokens).decode("utf-8", errors="ignore")

    def _morphological_decode(self, indices: list[int]) -> list[int]:
        indices = self.merge_rewriter.rewrite_bytes(indices, reverse=True)
        indices = self.decompose_rewriter.rewrite_bytes(indices, reverse=True)
        return indices


class MYTETokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
        self.encoder = MYTEEncoder()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256, "Vocabulary size should be at least 256 for MYTE-BPE."
        num_merges = vocab_size - 256

        if verbose:
            print("Applying MYTE encoding...")

        text_bytes = self.encoder.encode(text)
        ids = list(text_bytes)

        merges = {}
        vocab = {idx: [int(idx)] for idx in range(256)}  # int -> bytes
        if verbose:
            print("Applying merges...")
        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f"Merge {i+1}/{num_merges}: {pair} -> {idx} ({self.decode(vocab[idx])}) had {stats[pair]} occurrences")

        self.merges = merges
        self.vocab = vocab


    def encode(self, text):
        text_bytes = self.encoder.encode(text)
        ids = list(text_bytes)

        while len(ids) >= 2:
            stats = get_stats(ids)
            if not stats:
                break

            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break

            idx = self.merges[pair]
            ids = merge(ids, pair, idx)

        return ids

    def render_token(self, t: int) -> str:
        s = bytes([t]).decode('utf-8', errors='replace')
        return self.replace_control_characters(s)

    def replace_control_characters(self, s: str) -> str:
        chars = []
        for ch in s:
            if unicodedata.category(ch)[0] != "C":
                chars.append(ch)
            else:
                chars.append(f"\\u{ord(ch):04x}")
        return "".join(chars)

    def decode(self, ids):
        for idx in ids:
            if idx not in self.vocab:
                raise ValueError(f"Invalid token id: {idx}")

        def flatten(xss):
            return [x for xs in xss for x in xs]

        text_bytes = [self.vocab[idx] for idx in ids]
        return self.encoder.decode(flatten(text_bytes))

    def save(self, file_prefix):
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                s = self.render_token(token[0])
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = self.render_token(self.vocab[idx0][0])
                    s1 = self.render_token(self.vocab[idx1][0])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")
