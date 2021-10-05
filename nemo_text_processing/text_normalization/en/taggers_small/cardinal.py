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


from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_UPPER,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.en.taggers.cardinal import CardinalFst as defaultCardinalFst
from nemo_text_processing.text_normalization.en.utils import get_abs_path, load_labels
from pynini.lib.rewrite import top_rewrite

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g. 
        -23 -> cardinal { negative: "true"  integer: "twenty three" } }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        single_digits_graph = pynini.invert(graph_digit | graph_zero)
        self.single_digits_graph = single_digits_graph + pynini.closure(insert_space + single_digits_graph)
        self.optional_minus = pynini.closure(pynini.accep("-"), 0, 1)
        self.optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        self.filter = self.optional_minus + pynini.union(
            NEMO_DIGIT ** (5, ...), NEMO_DIGIT ** (2, 3) + pynini.closure(pynini.cross(",", "") + NEMO_DIGIT ** (3), 1)
        )

        self.filter |= self.optional_minus + pynini.union(
            NEMO_DIGIT ** (5, ...), NEMO_DIGIT ** (2, 3) + pynini.closure(pynini.cross(" ", "") + NEMO_DIGIT ** (3), 1)
        )

        self.graph = pynini.compose(self.filter.optimize(), self.optional_minus + self.single_digits_graph)

        final_graph = self.graph.optimize() | self.get_serial_graph().optimize() | self.get_roman_graph().optimize()
        final_graph = (
            self.optional_minus_graph + pynutil.insert("integer: \"") + final_graph.optimize() + pynutil.insert("\"")
        )
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

        assert top_rewrite("123,454", self.graph) == 'one two three four five four'

    def get_serial_graph(self):
        """
        Finite state transducer for classifying serial.
            The serial is a combination of digits, letters and dashes, e.g.:
            c325-b -> "c three two five b"
        """
        alpha = NEMO_UPPER
        num_graph = self.single_digits_graph

        delimiter = insert_space | pynini.cross("-", " ")
        letter_num = pynini.closure(alpha + delimiter, 1) + num_graph
        num_letter = pynini.closure(num_graph + delimiter, 1) + alpha
        num_delimiter_num = pynini.closure(num_graph + delimiter, 1) + num_graph
        next_alpha_or_num = pynini.closure(delimiter + (alpha | num_graph))
        serial_graph = (letter_num | num_letter | num_delimiter_num) + next_alpha_or_num
        serial_graph = pynini.compose(NEMO_SIGMA + NEMO_UPPER + NEMO_SIGMA, serial_graph)
        serial_graph.optimize()
        return pynutil.add_weight(serial_graph, 10)

    def get_roman_graph(self):
        """
        Finite state transducer for classifying roman numbers (only upper case is considered, "I" is excluded).
        e.g.:
            IV -> "four"
        """

        def _load_roman(file: str):
            roman = load_labels(get_abs_path(file))
            roman_numerals = [(x.upper(), y) for x, y in roman]
            return pynini.string_map(roman_numerals)

        cardinal_graph = defaultCardinalFst(deterministic=True).graph
        digit_teen = _load_roman("data/roman/digit_teen.tsv") @ cardinal_graph
        ties = _load_roman("data/roman/ties.tsv") @ cardinal_graph
        hundreds = _load_roman("data/roman/hundreds.tsv") @ cardinal_graph

        graph = (
            (ties | digit_teen | hundreds)
            | (ties + insert_space + digit_teen)
            | (hundreds + pynini.closure(insert_space + ties, 0, 1) + pynini.closure(insert_space + digit_teen, 0, 1))
        ).optimize()

        graph = graph + pynini.closure(pynutil.delete("."), 0, 1)
        graph = pynini.compose(NEMO_SIGMA - "I", graph)
        return graph
