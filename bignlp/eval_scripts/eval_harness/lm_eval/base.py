import abc
import enum
import random
import re
from functools import partial

import spacy
import numpy as np

from lm_eval.metrics import mean, perplexity, weighted_perplexity, weighted_mean


def _SPACY_NLP(*args, **kwargs):
    global _SPACY_NLP
    nlp = spacy.load("en_core_web_sm")
    _SPACY_NLP = nlp
    return nlp(*args, **kwargs)


class LM(abc.ABC):
    def __init__(self):
        self.cache_hook = CacheHook(None)

    @abc.abstractmethod
    def loglikelihood(self, requests):
        """Compute log-likelihood of generating a continuation from a context.
        Downstream tasks should attempt to use loglikelihoodikelihood instead of other
        LM calls whenever possible.

        :param requests: list
            A list of pairs (context, continuation)
            context: str
                Context string. Implementations of LM must be able to handle an
                empty context string.
            continuation: str
                The continuation over which log likelihood will be calculated. If
                there is a word boundary, the space should be in the continuation.
                For example, context="hello" continuation=" world" is correct.
        :return: list
            A list of pairs (logprob, isgreedy)
            logprob: float
                The log probability of `continuation`
            isgreedy:
                Whether `continuation` would be generated by greedy sampling from `context`
        """
        pass

    @abc.abstractmethod
    def loglikelihood_rolling(self, requests):
        """Compute full log-likelihood of a string, with no truncation, for perplexity computation
        - We will use the full max context length of the model.
        - For inputs that exceed the max context length, we divide the tokenized string into chunks of up to
        the max context length.
        - IMPORTANT: Each document's loglikelihood/perplexity is computed *separately*, unlike other implementations
          which may simply concatenate multiple documents together.
        - IMPORTANT: We maximize the amount of context for each prediction. Specifically, for inputs that we break into
          multiple chunks, the last input will still a full-sized context.
          Example:
            Input tokens: [ 0 1 2 3 4 5 6 7 8 9 ]
            Prefix: EOT
            Max context length: 4
            Resulting input/prediction pairs:

                INPUT:  EOT   0   1   2
                PRED:     0   1   2   3

                INPUT:    3   4   5   6
                PRED:     4   5   6   7

                INPUT:    5   6   7   8
                PRED:             8   9

          Observe that:
            1. Each token is predicted exactly once
            2. For the last pair, we provide the full context, but only score the last two tokens

        :param requests: list
            A list of strings
            string: str
                String for which we are computing per-toke  loglikelihood
        :return: list
            A list of pairs (logprob, isgreedy)
            logprob: float
                The log probability of `continuation`
            isgreedy:
                Whether `continuation` would be generated by greedy sampling from `context`
        """
        pass

    # TODO: Add an optional max length
    @abc.abstractmethod
    def greedy_until(self, requests):
        """Generate greedily until a stopping sequence

        :param requests: list
            A list of pairs (context, until)
            context: str
                Context string
            until: [str]
                The string sequences to generate until. These string sequences
                may each span across multiple tokens, or may be part of one token.
        :return: list
            A list of strings continuation
            continuation: str
                The generated continuation.
        """
        pass

    @classmethod
    def create_from_arg_string(cls, arg_string, *args, **kwargs):
        """Constructor method, in case models need additional arguments
        e.g. OpenAI API engine, paths for loading, other params

        :param arg_string: str
            Left up to individual model class to handle

        """
        return cls()

    def set_cache_hook(self, cache_hook):
        self.cache_hook = cache_hook

    @abc.abstractmethod
    def can_access_output(self):
        """
        Megatron model may use pipeline parallelism. In this case only the last GPU in a pipeline has access to the actual outputs.
        We need to check for this and only do metrics computation on processes that can actually access results.
        """
        pass


class ResultPreprocessing(enum.Enum):
    NONE = enum.auto()
    FIRST_SENTENCE = enum.auto()


class Task(abc.ABC):
    """A task represents an entire benchmark including its dataset, problems,
    answers, and evaluation methods. See BoolQ for a simple example implementation

    A `doc` can be any python object which represents one instance of evaluation.
    This is usually a dictionary e.g.
        {"question": ..., "answer": ...} or
        {"question": ..., question, answer)
    """

    RESULT_PREPROCESSING = ResultPreprocessing.NONE

    def __init__(self):
        self.download()
        self._training_docs = None
        self._fewshot_docs = None

    def download(self):
        """Downloads the task dataset if necessary"""
        pass

    @abc.abstractmethod
    def has_training_docs(self):
        """Whether the task has a training set"""
        pass

    @abc.abstractmethod
    def has_validation_docs(self):
        """Whether the task has a validation set"""
        pass

    @abc.abstractmethod
    def has_test_docs(self):
        """Whether the task has a test set"""
        pass

    def training_docs(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return []

    def validation_docs(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return []

    def test_docs(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return []

    def sample_examples(self, examples, k, rnd, **kwargs):
        """Sample k examples out of the iterable `examples`, using provided random number generator.

        :param examples: iterable of tuples (doc_id, doc): iterable of shot examples
        :param k: number of examples to be included in the prompt
        :param rnd: initialized random number generator object, e.g. rnd = random.Random(1337)
        :return: iterable of tuples (doc_id, doc): iterable of k shot examples
        """
        return rnd.sample(examples, k)

    def fewshot_examples(self, k, rnd, filter_func=None, **kwargs):
        """
        Draw k samples from the training dataset using random generator `rnd`.
        If `filter_func` is provided, it will be used to filter examples/documents (keep or exclude from sampling)
        
        :param k: number of examples to be included in the prompt
        :param rnd: initialized random number generator object, e.g. rnd = random.Random(1337)
        :param filter_func: function taking an iterable and returning an iterable a potentially smaller, filtered iterable

        :return: iterable of tuples (doc_id, doc): iterable of shot examples
        """
        if self._training_docs is None:
            self._training_docs = list(self.training_docs())
            self._examples = list(zip(range(len(self._training_docs)),
                                      self._training_docs))  # NOTE: compute each time if necessary to save memory

        if filter_func is not None:
            examples = filter_func(self._examples)
        else:
            examples = self._examples

        return self.sample_examples(examples, k, rnd, **kwargs)

    @abc.abstractmethod
    def doc_to_text(self, doc):
        pass

    @abc.abstractmethod
    def doc_to_target(self, doc):
        pass

    @abc.abstractmethod
    def construct_requests(self, doc, ctx):
        """ Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.
        Each document has as many requests as loglikelihoods to be calculated (multiple choice questions).

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        pass

    def preprocess_results(self, mode, results):
        """Preprocesses results based on some preprocessing mode.

        :param mode:
            The preprocessing mode, an enum value from ResultPreprocessing.
        :param results:
            The results of the requests created in construct_requests.
        """
        if mode not in ResultPreprocessing:
            raise ValueError(
                f'Invalid mode, expected type {ResultPreprocessing.__name__} but got {type(mode).__name__}')

        if mode is ResultPreprocessing.NONE:
            preprocessed = results
        elif mode is ResultPreprocessing.FIRST_SENTENCE:
            preprocessed = []
            for result in results:
                if result:
                    spacy_doc = _SPACY_NLP(result)
                    preprocessed.append(str(next(spacy_doc.sents)))
                else:
                    preprocessed.append(result)
        else:
            raise RuntimeError(f'Unimplemented mode: {mode}')

        return preprocessed

    def compute_doc_metrics(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document.

        Results are initially preprocessed based on the value of the class attribute
        `RESULT_PREPROCESSING` before being passed to `process_results`.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        results = self.preprocess_results(self.RESULT_PREPROCESSING, results)
        return self.process_results(doc, results)

    @abc.abstractmethod
    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        pass

    @abc.abstractmethod
    def aggregation(self):
        """
        :returns: {str: [metric_score] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metric scores
        """
        pass

    @abc.abstractmethod
    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        pass

    def fewshot_description(self):
        return ""

    def filter_shots(self, shot_examples, doc):
        """
        Selectively keep only part of all possible shot examples,
        potentially based on characteristics of the document under examination.

        :param shot_examples: iterable of tuples (doc_id, doc)): samples to be used as shots
        :param doc: sample, document under examination

        :return: iterable of tuples (doc_id, doc): filtered iterable of shot examples
        """
        raise (NotImplementedError(
            '`filter_shots` must be implemented in child Task in order to use `filter_shot_examples=True`'))

    def fewshot_context(self, doc, num_fewshot, provide_description, rnd, filter_shot_examples=False, **kwargs):
        """Construct and format full prompt string for a given sample, optionally including description and shot examples
        :param doc: document object corresponding to the sample under examination 
        :param num_fewshot: number of examples to be included in the prompt
        :param provide_description: (bool), whether to prepend natural language description
        :param rnd: initialized random number generator object, e.g. rnd = random.Random(1337)
        :param filter_shot_examples: If True, will make sure to exclude certain samples from the prompt context, based
            on member `filter_shots` function
        :return: (shot_ids, context_str): tuple of (iterable of shot example IDs, string correspoding to context/prompt)
        """

        raw_description = self.fewshot_description()
        description = (raw_description + "\n===\n\n") if provide_description and raw_description else ""

        if num_fewshot == 0:
            labeled_examples = ""
            shot_ids = []
        else:
            # for sets with no training docs, draw from other set *but ensure no overlap with current doc*
            if self.has_training_docs():
                if filter_shot_examples:
                    fewshotex = self.fewshot_examples(k=num_fewshot, rnd=rnd,
                                                      filter_func=partial(self.filter_shots, doc=doc), **kwargs)
                else:
                    fewshotex = self.fewshot_examples(k=num_fewshot, rnd=rnd, **kwargs)
            else:
                if self._fewshot_docs is None:
                    self._fewshot_docs = list(
                        self.validation_docs() if self.has_validation_docs() else self.test_docs())
                    self._fewshot_docs = list(zip(range(len(self._fewshot_docs)), self._fewshot_docs))
                if filter_shot_examples:
                    fewshotex = self.filter_shots(self._fewshot_docs, doc)
                else:
                    fewshotex = self._fewshot_docs

                fewshotex = self.sample_examples(fewshotex, num_fewshot + 1, rnd, **kwargs)

                # get rid of the doc that's the one we're evaluating, if it's in the fewshot
                # works because dictionary-like objects support equality operation in Python
                fewshotex = [x for x in fewshotex if x != doc][:num_fewshot]

            shot_ids, shot_docs = zip(*fewshotex)
            labeled_examples = "\n\n".join(
                [self.doc_to_text(doc) + self.doc_to_target(doc) for doc in shot_docs]
            ) + "\n\n"

        example = self.doc_to_text(doc)  # the document of interest, main part of the prompt
        prompt_str = description + labeled_examples + example  # the formatted prompt string
        return shot_ids, prompt_str


class MultipleChoiceTask(Task):
    def doc_to_target(self, doc):
        return " " + doc['choices'][doc['gold']]

    def construct_requests(self, doc, ctx):
        lls = [
                  rf.loglikelihood(ctx, " {}".format(choice))[0]
                  for choice in doc['choices']  # get likelihoods
              ] + [
                  rf.loglikelihood(ctx, " {}".format(choice))[2]
                  for choice in doc['choices']  # get tokens
              ]
        return lls

    def process_results(self, doc, results):
        gold = doc["gold"]
        num_choices = len(doc["choices"])
        logprobs = results[:num_choices]

        choice_tokens = results[num_choices:]
        assert len(logprobs) == len(choice_tokens)
        normed_logprobs = [lp / len(x) for lp, x in zip(logprobs, choice_tokens)]
        acc = 1. if np.argmax(logprobs) == gold else 0.
        acc_norm = 1. if np.argmax(normed_logprobs) == gold else 0

        # NOTE(zhunliu): the previous normed setting is not ideal, norm by char
        # completion_len = np.array([float(len(i)) for i in doc["choices"]])
        # acc_norm = 1. if np.argmax(results / completion_len) == gold else 0.

        return {
            "acc": acc,
            "acc_norm": acc_norm,
        }

    def serialize_results(self, doc, results):
        num_choices = len(doc["choices"])
        logprobs = results[:num_choices]
        choice_tokens = results[num_choices:]
        assert len(logprobs) == len(choice_tokens)
        pred = np.argmax(logprobs)
        normed_logprobs = [lp / len(x) for lp, x in zip(logprobs, choice_tokens)]
        pred_norm = np.argmax(normed_logprobs)

        model_choice = doc["choices"][pred]
        model_choice_norm = doc["choices"][pred_norm]
        gold_choice = doc["choices"][doc["gold"]]
        return {
            "format": self.doc_to_text(doc) + " {choices}",
            "model_choice": model_choice,
            "model_choice_norm": model_choice_norm,
            "gold_choice": gold_choice,
            "choices": dict(zip(doc["choices"], results[:num_choices])),
        }

    def higher_is_better(self):
        return {
            "acc": True,
            "acc_norm": True,
        }

    def aggregation(self):
        return {
            "acc": mean,
            "acc_norm": mean,
        }

    def get_answer_ctx(self):
        """Return the answer-prompting string for the question.
           Most QA tasks has the format of "Question: xxx\nAnswer: "
           In this case the answer-prompting string is "Answer: "
        """
        raise NotImplementedError


class PerplexityTask(Task, abc.ABC):

    def has_training_docs(self):
        return False

    def fewshot_description(self):
        return ""

    def fewshot_examples(self, k, rnd):
        assert k == 0
        return []

    def fewshot_context(self, doc, num_fewshot, provide_description, rnd, **kwargs):
        assert num_fewshot == 0
        assert not provide_description
        return ([], "")

    def higher_is_better(self):
        return {
            "word_perplexity": False,
            "byte_perplexity": False,
            "bits_per_byte": False,
        }

    def doc_to_text(self, doc):
        return ""

    def doc_to_target(self, doc):
        return doc

    def construct_requests(self, doc, ctx):
        assert not ctx
        req = rf.loglikelihood_rolling(self.doc_to_target(doc))
        return req

    def process_results(self, doc, results):
        loglikelihood, = results
        words = self.count_words(doc)
        bytes = self.count_bytes(doc)
        return {
            "word_perplexity": (loglikelihood, words),
            "byte_perplexity": (loglikelihood, bytes),
            "bits_per_byte": (-loglikelihood, self.count_bytes(doc))
        }

    def aggregation(self):
        return {
            "word_perplexity": weighted_perplexity,
            "byte_perplexity": weighted_perplexity,
            "bits_per_byte": weighted_mean
        }

    def count_bytes(self, doc):
        return len(doc.encode("utf-8"))

    def count_words(self, doc):
        """ Downstream tasks with custom word boundaries should override this! """
        return len(re.split(r"\s+", doc))


req_ret_lens = {
    'loglikelihood': 4,
    'greedy_until': None,
    'loglikelihood_rolling': None,
}

import os
import json
import hashlib
from sqlitedict import SqliteDict


def hash_args(attr, args):
    dat = json.dumps([attr] + list(args))
    return hashlib.sha256(dat.encode('utf-8')).hexdigest()


class CacheHook:
    def __init__(self, cachinglm):
        if cachinglm is None:
            self.dbdict = None
            return

        self.dbdict = cachinglm.dbdict

    def add_partial(self, attr, req, res):
        if self.dbdict is None:
            return
        hsh = hash_args(attr, req)
        self.dbdict[hsh] = res


class CachingLM:
    def __init__(self, lm, cache_db):
        self.lm = lm
        self.cache_db = cache_db
        if os.path.dirname(cache_db): os.makedirs(os.path.dirname(cache_db), exist_ok=True)
        self.dbdict = SqliteDict(cache_db, autocommit=True)

        # add hook to lm
        lm.set_cache_hook(self.get_cache_hook())

    def __getattr__(self, attr):
        def fn(requests):
            res = []
            remaining_reqs = []

            # figure out which ones are cached and which ones are new
            for req in requests:
                hsh = hash_args(attr, req)
                if hsh in self.dbdict:
                    ob = self.dbdict[hsh]

                    assert ob is not None

                    res.append(ob)
                else:
                    res.append(None)
                    remaining_reqs.append(req)

            # actually run the LM
            rem_res = getattr(self.lm, attr)(remaining_reqs)

            # stick the new ones back into the list and also cache any of the new ones
            resptr = 0
            for req, r in zip(remaining_reqs, rem_res):
                while res[resptr] is not None: resptr += 1

                res[resptr] = r

                # caching
                hsh = hash_args(attr, req)
                self.dbdict[hsh] = r
            self.dbdict.commit()

            return res

        return fn

    def get_cache_hook(self):
        return CacheHook(self)


class Request:
    def __init__(self, type, args, index=None):
        if type not in req_ret_lens.keys():
            raise NotImplementedError('The request type {} is not implemented!'.format(type))

        self.type = type
        self.args = args
        self.index = index

    def __iter__(self):
        if req_ret_lens[self.type] is None:
            raise IndexError('This request type does not return multiple arguments!')
        i = 0
        for i in range(req_ret_lens[self.type]):
            yield Request(self.type, self.args, i)

    def __getitem__(self, i):
        if req_ret_lens[self.type] is None:
            raise IndexError('This request type does not return multiple arguments!')
        return Request(self.type, self.args, i)

    def __eq__(self, other):
        return self.type == other.type and self.args == other.args and self.index == other.index

    def __repr__(self):
        return f"Req_{self.type}{self.args}[{self.index}]\n"


class RequestFactory:
    def __getattr__(self, attr):
        def fn(*args):
            return Request(attr, args)

        return fn


rf = RequestFactory()