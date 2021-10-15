# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, NEMO_DIGIT, GraphFst, delete_space, insert_space
from nemo_text_processing.text_normalization.en.utils import get_abs_path

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class ElectronicFst(GraphFst):
    """
    Finite state transducer for verbalizing electronic
        e.g. tokens { electronic { username: "cdf1" domain: "abc.edu" } } -> c d f one at a b c dot e d u

    Args:
        deterministic: if True will provide a single transduction option,
        for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="verbalize", deterministic=deterministic)
        graph_digit_no_zero = pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv"))).optimize()
        graph_zero = pynini.cross("0", "zero")

        graph_digit = graph_digit_no_zero | graph_zero
        graph_symbols = pynini.string_file(get_abs_path("data/electronic/symbols.tsv")).optimize()
        user_name = (
            pynutil.delete("username:")
            + delete_space
            + pynutil.delete("\"")
            + (
                pynini.closure(
                    pynutil.add_weight(graph_digit + insert_space, 1.09)
                    | pynutil.add_weight(pynini.closure(graph_symbols + insert_space), 1.09)
                    | pynutil.add_weight(NEMO_NOT_QUOTE + insert_space, 1.1)
                )
            )
            + pynutil.delete("\"")
        )

        server_common = pynini.string_file(get_abs_path("data/electronic/server_name.tsv"))
        domain_common = pynini.string_file(get_abs_path("data/electronic/domain.tsv"))

        convert_defaults = (
            NEMO_NOT_QUOTE | pynutil.add_weight(domain_common, -0.1) | pynutil.add_weight(server_common, -0.1)
        )
        domain = convert_defaults + pynini.closure(pynutil.insert(" ") + convert_defaults)
        domain = pynini.compose(
            domain,
            pynini.closure(
                pynutil.add_weight(graph_symbols, -0.1) | pynutil.add_weight(graph_digit, -0.1) | NEMO_NOT_QUOTE
            ),
        )

        domain = (
            pynutil.delete("domain:")
            + delete_space
            + pynutil.delete("\"")
            + domain
            + delete_space
            + pynutil.delete("\"")
        )


        # urls
        protocol = pynutil.delete("protocol: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        
        
        # ids
        # 2.7.7.68 -> two dot seven seven six eight
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        single_digits_graph = pynini.invert(graph_digit | graph_zero)
        single_digits_graph = single_digits_graph + pynini.closure(insert_space + single_digits_graph)
        dot = pynini.cross(".", " dot ")
        id = single_digits_graph + dot + single_digits_graph + dot + single_digits_graph + dot + single_digits_graph
        
        
        
        graph = (
            pynini.closure(protocol + delete_space, 0, 1)
            + pynini.closure(user_name + delete_space + pynutil.insert("at ") + delete_space, 0, 1)
            + domain
        )

        graph |= pynutil.delete("protocol: \"") + id + pynutil.delete("\"")

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
