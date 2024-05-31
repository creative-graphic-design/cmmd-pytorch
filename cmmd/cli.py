# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The main entry point for the CMMD calculation."""

import argparse
from typing import Optional

from cmmd.cmmd_score import compute_cmmd


def parse_args(prog: Optional[str] = None):
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument(
        "ref_dir", type=str, help="Path to the directory containing reference images."
    )
    parser.add_argument(
        "eval_dir",
        type=str,
        help="Path to the directory containing images to be evaluated.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation.",
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=-1,
        help="Maximum number of images to read from each directory.",
    )
    parser.add_argument(
        "--ref-embed-file",
        type=str,
        help="Path to the pre-computed embedding file for the reference images.",
    )
    return parser.parse_args()


def run():
    args = parse_args()

    cmmd_score = compute_cmmd(
        ref_dir=args.ref_dir,
        eval_dir=args.eval_dir,
        ref_embed_file=args.ref_embed_file,
        batch_size=args.batch_size,
        max_count=args.max_count,
    )
    print(f"The CMMD value is: {cmmd_score:.3f}")
