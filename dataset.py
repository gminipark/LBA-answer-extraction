from torch.utils.data import Dataset
import torch

class AnswerExtractionDataset(Dataset):

    def __init__(self, tokenizer, examples, max_length=384, stride=128, pad_to_max_length=False):

        self.tokenizer = tokenizer
        self.examples = examples
        self.max_length = max_length
        self.doc_stride = stride
        self.pad_on_right = tokenizer.padding_side == "right"
        self.pad_to_max_length = pad_to_max_length
        self.features = self.examples_to_features()

    def examples_to_features(self):
        questions = [example['question'] for example in self.examples]
        answers = [example['answer'] for example in self.examples]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            questions if self.pad_on_right else answers,
            answers if self.pad_on_right else questions,
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=False,
            padding="max_length" if self.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        # tokenized_examples["example_id"] = []

        # for i in range(len(tokenized_examples["input_ids"])):
        #     # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        #     sequence_ids = tokenized_examples.sequence_ids(i)
        #     context_index = 1 if self.pad_on_right else 0

        #     # One example can give several spans, this is the index of the example containing this span of text.
        #     sample_index = sample_mapping[i]
        #     #tokenized_examples["example_id"].append(examples["id"][sample_index])

        #     # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        #     # position is part of the context or not.
        #     tokenized_examples["offset_mapping"][i] = [
        #         (o if sequence_ids[k] == context_index else None)
        #         for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        #     ]

        return tokenized_examples


    def __getitem__(self, idx):

        return {key : torch.tensor(self.features[key][idx],dtype=torch.long) for key in self.features.keys()} 

    def __len__(self):
        return len(self.examples)