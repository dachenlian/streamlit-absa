from datetime import datetime
from pathlib import Path
import re

from bs4 import Tag

from scrape.types import PTTComment, PTTMetadata, PTTPost


def get_ptt_post_metadata(soup: Tag) -> PTTMetadata:
    try:
        author_ip, url = soup.select("span.f2")
    except ValueError:
        author_ip, url = "", ""
    else:
        author_ip = author_ip.get_text(strip=True)
        url = url.get_text(strip=True).split("網址:")[-1]

    main_content_soup = soup.select_one("#main-content").contents

    author = main_content_soup[0].contents[1].get_text(strip=True)
    board = main_content_soup[1].contents[1].get_text(strip=True)
    title = main_content_soup[2].contents[1].get_text(strip=True)
    published_date = main_content_soup[3].contents[1].get_text(strip=True)
    published_date = datetime.strptime(published_date, "%a %b %d %H:%M:%S %Y")

    metadata = PTTMetadata(
        author=author,
        board=board,
        title=title,
        published_date=published_date,
        author_ip=author_ip,
        url=url,
    )

    return metadata


def get_ptt_comments(soup: Tag) -> list[PTTComment]:
    from scrape.utils import select_tag
    comments = []
    for tag in soup.select(".push:not(.warning-box)"):
        push_tag = select_tag(tag, ".push-tag")
        username = select_tag(tag, ".push-userid")
        content = select_tag(tag, ".push-content")[1:]
        published_date = select_tag(tag, ".push-ipdatetime")
        published_date = datetime.strptime(published_date, "%m/%d %H:%M")
        comment = PTTComment(
            tag=push_tag,  # type: ignore
            username=username,
            content=content,
            published_date=published_date,
        )
        comments.append(comment)

    return comments


def get_ptt_post(path: str | Path) -> PTTPost:
    from scrape.utils import make_soup

    soup = make_soup(str(path))
    main_body = []
    metadata = get_ptt_post_metadata(soup)
    main_content_soup = soup.select_one("#main-content").contents
    end = soup.select_one("span.f2")

    main_content_soup = main_content_soup[4:]

    for tag in main_content_soup:
        if tag == end:
            main_body[-1] = main_body[-1].strip("\n-- ")
            break
        main_body.append(tag.get_text(strip=True))

    main_body = "\n".join(main_body)

    comments = get_ptt_comments(soup)

    post = PTTPost(
        metadata=metadata,
        main_body=main_body,
        comments=comments,
    )
    return post
