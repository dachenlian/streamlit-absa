from pydantic import BaseModel


from collections import Counter


class CountABSAOutput(BaseModel):
    positive: Counter
    negative: Counter
    neutral: Counter


GetABSAOutput = list[tuple[str, dict[str, str]]]
GetAnnotatedABSAOutput = list[str | tuple[str, str, str]]