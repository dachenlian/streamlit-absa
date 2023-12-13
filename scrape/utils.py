from dataclasses import asdict
import json
from pathlib import Path

from bs4 import BeautifulSoup, Tag
from dotenv import load_dotenv, find_dotenv
import jieba
import openai
import pandas as pd
import re
import requests

from scrape.imdb import get_imdb_reviews
from scrape.letterboxd import get_letterboxd_reviews
from scrape.ptt import get_ptt_post
from scrape.types import GetDataOutput, OpenAIModel
from scrape.analysis import create_word_count, get_summary
from scrape.absa.types import GetABSAOutput, GetAnnotatedABSAOutput
from scrape.absa import aspects
from scrape.absa.prompts import GET_ABSA_MOVIE_PROMPT, GET_ABSA_FINANCE_PROMPT
from scrape.prompts import MOVIE_SUMMARY_PROMPT, FINANCE_SUMMARY_PROMPT

# load_dotenv(find_dotenv())

BASE = Path(__file__).resolve().parent.parent


def make_soup(url: str, headers: dict | None = None) -> BeautifulSoup:
    if Path(url[:500]).exists():
        with open(url, "r") as f:
            soup = BeautifulSoup(f.read(), "lxml")
            return soup

    if not url.startswith("http"):
        url = "https://" + url

    if headers is None:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:47.0) Gecko/20100101 Firefox/47.0",
            "Accept-Language": "en-US;q=0.8,zh-TW;q=0.2",
        }
    else:
        headers[
            "User-Agent"
        ] = "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:47.0) Gecko/20100101 Firefox/47.0"
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "lxml")
    return soup


def chunker(seq: list[str], size: int) -> list[list[str]]:
    return [seq[pos : pos + size] for pos in range(0, len(seq), size)]


def select_tag(soup: Tag, selector: str) -> str:
    tag = soup.select_one(selector)
    if not tag:
        return ""
    return tag.get_text(strip=True)


def get_data(
    client: openai.Client,
    url: str,
    max_length: int = 2000,
) -> GetDataOutput:
    from scrape.absa.absa import (
        create_absa_counts_df,
        get_absa,
        count_absa,
    )

    if ("letterboxd" in url) or ("imdb" in url):
        if "letterboxd" in url:
            getter = get_letterboxd_reviews
        else:
            getter = get_imdb_reviews
        meta, output = getter(url)

        with open(BASE / "data/stopwords.txt", "r") as f:
            stop = f.read().splitlines()
        title = meta.movie_title
        source = "movie"
        samples = [asdict(r) for r in output]
        texts = [r.review for r in output]
        summary = get_summary(
            client,
            texts,
            base_prompt=MOVIE_SUMMARY_PROMPT,
            max_length=max_length,
            model_name=OpenAIModel.GPT3_5,
        )
        absa = get_absa(
            client,
            texts,
            max_length=max_length,
            base_prompt=GET_ABSA_MOVIE_PROMPT,
            model_name=OpenAIModel.GPT3_5,
        )
        df = pd.DataFrame(samples)
        df = prepare_movie_df(absa, df)
        df_filter = filter_movie_df
        aspect_list = aspects.MOVIE_ASPECTS
    elif "ptt" in url:
        jieba.set_dictionary(BASE / "data/dict.txt.big")
        output = get_ptt_post(url)
        title = output.metadata.title
        source = "ptt"
        samples = [asdict(c) for c in output.comments]
        texts = [c.content for c in output.comments]
        texts = [re.sub(r"\s+", " ", " ".join(jieba.lcut(t)).strip()) for t in texts]

        comments = [f"{c.username}: {c.content}" for c in output.comments]
        main_body = f"Poster: {output.metadata.author}\n{output.main_body}"
        summary = get_summary(
            client,
            texts=comments,
            main_body=main_body,
            max_length=max_length,
            base_prompt=FINANCE_SUMMARY_PROMPT,
            model_name=OpenAIModel.GPT3_5,
        )
        absa = get_absa(
            client,
            texts,
            max_length=max_length,
            base_prompt=GET_ABSA_FINANCE_PROMPT,
            model_name=OpenAIModel.GPT3_5,
        )
        with open(BASE / "data/trad-stopwords.txt", "r") as f:
            stop = f.read().splitlines()
        aspect_list = aspects.FINANCIAL_ASPECTS
        df_filter = None
        df = pd.DataFrame(samples)
    else:
        raise ValueError(f"Unknown website: {url}")

    absa_counts = count_absa(absa)
    absa_counts_df = create_absa_counts_df(absa_counts, proportional=True)
    counts = create_word_count(texts, stop)

    return GetDataOutput(
        title=title,
        source=source,
        samples=samples,
        word_counts=counts,
        summary=summary,
        df=df,
        absa_counts_df=absa_counts_df,
        df_filter=df_filter,
        aspect_list=aspect_list,
    )


def prepare_movie_df(absa: GetABSAOutput, df: pd.DataFrame) -> pd.DataFrame:
    from scrape.absa.absa import create_absa_df, get_val_from_absa_output_key

    contains_spoilers = get_val_from_absa_output_key(absa, "contains_spoilers")
    df["contains_spoilers"] = contains_spoilers
    df["contains_spoilers"].fillna(True).map({True: "Yes", False: "No"})
    df["rating"] = df["rating"].fillna(-1)
    absa_df = create_absa_df(absa)
    df = pd.merge(df, absa_df, left_index=True, right_index=True, how="outer")
    return df


def filter_movie_df(
    hide_spoilers: bool,
    hide_negative: bool,
    hide_positive: bool,
    hide_neutral: bool,
    df: pd.DataFrame,
) -> pd.DataFrame:
    if hide_spoilers:
        df = df[df["contains_spoilers"] == "No"]

    if hide_negative:
        df = df[~(df["negative"] != "")]

    if hide_positive:
        df = df[~(df["positive"] != "")]

    if hide_neutral:
        df = df[~(df["neutral"] != "")]

    return df


async def acall_model(
    client: openai.AsyncOpenAI,
    messages: list[dict[str, str]],
    model_name: OpenAIModel = OpenAIModel.GPT3_5,
    return_json: bool = False,
) -> str | dict | GetAnnotatedABSAOutput:
    if return_json:
        response_format = {"type": "json_object"}
    else:
        response_format = {"type": "text"}

    response = await client.chat.completions.create(
        messages=messages,
        model=model_name,  # gpt-4-1106-preview
        response_format=response_format,
    )

    response = response.choices[0].message.content
    assert response

    if return_json:
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            response = {}

    return response


def call_model(
    client: openai.Client,
    messages: list[dict[str, str]],
    model_name: OpenAIModel = OpenAIModel.GPT3_5,
    return_json: bool = False,
) -> str | dict | GetAnnotatedABSAOutput:
    if return_json:
        response_format = {"type": "json_object"}
    else:
        response_format = {"type": "text"}

    response = client.chat.completions.create(
        messages=messages,
        model=model_name,  # gpt-4-1106-preview
        response_format=response_format,
    )

    response = response.choices[0].message.content
    assert response

    if return_json:
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            response = {}

    return response


# def filter_df(df: pd.DataFrame, constraints: list[FilterDFConstraint]):
#     for c in constraints:
#         if c.name in df.columns:
