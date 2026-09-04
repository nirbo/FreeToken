from freetoken.message import DetokenizeMsg
from freetoken.tokenizer.detokenize import DetokenizeManager


class _Tokenizer:
    eos_token_id = 0

    def batch_decode(self, batches):
        pieces = {1: "one ", 2: "two "}
        return ["".join(pieces[token] for token in batch) for batch in batches]


def test_multiple_tokens_for_one_request_advance_offsets_in_order():
    manager = DetokenizeManager(_Tokenizer())
    output = manager.detokenize(
        [
            DetokenizeMsg(uid=7, next_token=1, finished=False),
            DetokenizeMsg(uid=7, next_token=2, finished=True),
        ]
    )
    assert "".join(output) == "one two "
