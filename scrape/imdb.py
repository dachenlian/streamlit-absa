import re
from bs4.element import Tag

from scrape.types import IMDbReview
from scrape.types import MovieMetadata


def get_imdb_reviews(url: str) -> tuple[MovieMetadata, list[IMDbReview]]:
    from scrape.utils import make_soup
    soup = make_soup(url)
    movie_title = soup.select_one("h3[itemprop='name']").get_text(strip=True)
    metadata = MovieMetadata(
        movie_title=movie_title
    )
    reviews = soup.select(".imdb-user-review")
    parsed = [get_imdb_review(r, metadata=metadata) for r in reviews]
    return metadata, parsed


def get_imdb_rating(r: Tag) -> int | None:
    rating_elm = r.select_one("span.point-scale")
    if not rating_elm:
        return None
    rating = int(rating_elm.previous_sibling.get_text(strip=True))
    return rating


def get_imdb_review(r: Tag, metadata: dict) -> IMDbReview:
    review_title = r.select_one("a.title").get_text(strip=True)
    rating = get_imdb_rating(r)
    username = r.select_one("span.display-name-link").get_text(strip=True)
    date = r.select_one("span.review-date").get_text(strip=True)
    review = r.select_one(".content .text").get_text(strip=True).strip()
    review = re.sub(r"\n+", "\n", review)
    helpful = r.select_one(".actions.text-muted").next_element.get_text(strip=True)

    return IMDbReview(
        username=username,
        date=date,
        review_title=review_title,
        rating=rating,
        review=review,
        helpful=helpful,
    )
