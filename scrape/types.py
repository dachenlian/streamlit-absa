from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Literal

import pandas as pd
from pydantic import BaseModel


@dataclass
class PTTComment:
    username: str
    tag: Literal["推", "噓", "→"]
    content: str
    published_date: datetime

    def __str__(self):
        out = ""
        for k, v in asdict(self).items():
            out += f"{k}: {v}\n"
        return out


@dataclass
class PTTMetadata:
    author: str
    board: str
    title: str
    published_date: datetime
    author_ip: str
    url: str


@dataclass
class PTTPost:
    metadata: PTTMetadata
    comments: list[PTTComment]
    main_body: str

    def __str__(self):
        out = ""
        for k, v in asdict(self.metadata).items():
            out += f"{k}: {v}\n"
        out += f"MAIN_BODY:\n{self.main_body}\n"
        comments = "\n".join([str(c) for c in self.comments])
        out += f"COMMENTS:\n{comments}\n"
        return out


@dataclass
class MovieMetadata:
    movie_title: str
    poster_url: str | None = None


@dataclass
class LetterboxdReview:
    username: str
    date: str
    rating: int | None
    review: str

    def __str__(self):
        out = ""
        for k, v in asdict(self).items():
            if k == "movie_title":
                continue
            out += f"{k}: {v}\n"
        return out


@dataclass
class IMDbReview:
    review_title: str
    rating: int | None
    username: str
    date: str
    review: str
    helpful: str

    def __str__(self):
        out = ""
        for k, v in asdict(self).items():
            if k == "movie_title":
                continue
            out += f"{k}: {v}\n"
        return out


class GetDataOutput(BaseModel, arbitrary_types_allowed=True):
    title: str
    source: Literal["movie", "ptt"]
    samples: list[dict]
    word_counts: Counter
    df: pd.DataFrame
    absa_counts_df: pd.DataFrame
    aspect_list: list[str]
    summary: str
    df_filter: Callable[..., pd.DataFrame] | None = None


class FilterDFConstraint(BaseModel):
    name: str
    value: Any


class OpenAIModel(str, Enum):
    GPT3_5 = "gpt-3.5-turbo-1106"
    GPT4 = "gpt-4-1106-preview"
