import re
import time

from bs4 import BeautifulSoup
from bs4.element import Tag
from loguru import logger
import requests
from tqdm.auto import tqdm

from scrape.types import LetterboxdReview
from scrape.types import MovieMetadata


def _get_letterboxd_rating(r: Tag) -> int | None:
    rating_elm = r.select_one("span.rating")
    if not rating_elm:
        return None

    classes = rating_elm["class"]
    rating = [c for c in classes if c.startswith("rated-")][0]
    rating = int(rating.replace("rated-", ""))

    return rating


def _get_letterboxd_review_text(r: Tag) -> str:
    """如果全文被砍掉了，就去抓全文的網址"""
    review_elm = r.select_one(".body-text")
    is_collapsed_text = bool(review_elm.select_one(".collapsed-text"))  # 代表全文被砍掉了
    if is_collapsed_text:
        attrs = review_elm.attrs
        review_url = attrs["data-full-text-url"]
        url = f"https://letterboxd.com{review_url}"
        soup = BeautifulSoup(requests.get(url).text, "lxml")
        review = soup.select_one("body").get_text(strip=True)
        return review
    review = review_elm.text.strip()
    review = re.sub(r"\n+", "\n", review)
    return review


def _get_letterboxd_review(r: Tag) -> LetterboxdReview:
    username = r.select_one("strong.name").get_text(strip=True)
    date = r.select_one("span._nobr").get_text(strip=True)
    rating = _get_letterboxd_rating(r)
    # likes = # 需要JS
    review = _get_letterboxd_review_text(r)

    return LetterboxdReview(
        username=username,
        date=date,
        rating=rating,
        review=review,
    )


def get_letterboxd_reviews(url: str) -> tuple[MovieMetadata, list[LetterboxdReview]]:
    from scrape.utils import make_soup

    soup = make_soup(url)
    movie_title = soup.select_one("h1.headline-2.prettify a").get_text(strip=True)
    poster_url = soup.select_one(".film-poster").attrs["data-poster-url"]
    poster_url = f"https://letterboxd.com{poster_url}"
    metadata = MovieMetadata(movie_title=movie_title, poster_url=poster_url)
    reviews = soup.select("li.film-detail")
    if not reviews:
        raise ValueError("No reviews found")

    parsed = [_get_letterboxd_review(r) for r in reviews]

    return metadata, parsed


def get_many_letterboxd_reviews(
    base_url: str, start_page: int = 1, stop_page: int = 20
) -> list[LetterboxdReview]:
    reviews = []
    for page in tqdm(range(start_page, stop_page + 1), desc="Page"):
        url = base_url.format(page=page)
        metadata, reviews_on_page = get_letterboxd_reviews(url)
        reviews.extend(reviews_on_page)
        # logger.info(f"Page {page}: got {len(reviews_on_page)} reviews")
        time.sleep(5)
    logger.info(f"Got {len(reviews)} reviews")
    return reviews
