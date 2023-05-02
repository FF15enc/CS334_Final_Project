from datetime import datetime
import re
import requests
import bs4
import pandas as pd
import numpy as np

"""
    name: str
    team: str  # might be different for the same team
    agent: str
    rating: float
    acs: int  # average combat score
    k: int  # kills
    d: int  # deaths
    a: int  # assists
    tkmd: int  # total kills minus deaths
    kast: float  # kill, assist, survive, trade %
    adr: int  # average damage per round
    hs: float  # headshot %
    fk: int  # first kills
    fd: int  # first deaths
    fkmd: int  # first kills minus first deaths
    clutch: float  # win clutches / total experience clutches, vlr.gg does not have this data but rib.gg has
"""
vlrgg_columns = ["match-datetime", "patch", "map", "team1", "team2", "team1-score", "team2-score",
                 "player-name", "player-team", "agent", "rating", "rating-t", "rating-ct", "acs", "acs-t", "acs-ct",
                 "k", "k-t", "k-ct", "d", "d-t", "d-ct", "a", "a-t", "a-ct", "tkmd", "tkmd-t", "tkmd-ct",
                 "kast", "kast-t", "kast-ct", "adr", "adr-t", "adr-ct", "hs", "hs-t", "hs-ct", "fk", "fk-t", "fk-ct",
                 "fd", "fd-t", "fd-ct", "fkmd", "fkmd-t", "fkmd-ct"]


def vlrgg_scraper(url: str) -> pd.DataFrame:
    """
    scraper using beautifulsoup4 and requests to collect player statistics from each match result page of vlr.gg
    :param url:
    :return:
    """
    page = requests.get(url)
    soup = bs4.BeautifulSoup(page.content, "html.parser")
    data_table = []

    # match stats (datetime, patch)
    match_stats = soup.find("div", {"class": "match-header-date"})
    datetime_stat = match_stats.find("div", {"class": "moment-tz-convert"}).get("data-utc-ts")
    patch_stat = match_stats.find("div", {"style": "margin-top: 4px;"}).find("div", {"style": "font-style: italic;"})
    match_datetime = datetime.strptime(datetime_stat, "%Y-%m-%d %H:%M:%S")
    patch = patch_stat.string.strip().replace("Patch ", "")

    map_stats = soup.find_all("div", {"class": "vm-stats-game"})
    map_stats = [map_stat for map_stat in map_stats if map_stat.get("data-game-id") != "all"]  # remove all game results

    for map_stat in map_stats:
        # team wise result for this map
        map_result = map_stat.find("div", {"class": "vm-stats-game-header"})
        map_name = map_result.find("div", {"class": "map"}).find("span", {"style": "position: relative;"}).contents[
            0].strip()
        team_stats = map_result.find_all("div", {"class": "team"})
        team1_stat, team2_stat = team_stats
        team1_name = team1_stat.find("div", {"class": "team-name"}).string.strip()
        team2_name = team2_stat.find("div", {"class": "team-name"}).string.strip()
        team1_score = team1_stat.find("div", {"class": "score"}).string
        team2_score = team2_stat.find("div", {"class": "score"}).string

        teams_player_stats = map_stat.find_all("table", "wf-table-inset")
        team1_player_stats, team2_player_stats = teams_player_stats

        def parse_player_stats(team_player_stats: bs4.element.Tag, team_name: str = None) -> None:
            team_player_stats_list = team_player_stats.find("tbody").find_all("tr")
            for player_stats in team_player_stats_list:
                agent = player_stats.find("td", {"class": "mod-agents"}).find("img").get("title")
                # form the data
                split_pattern = r"[\t\n]+"
                player_stats_list = [match_datetime, patch, map_name, team1_name, team2_name, team1_score, team2_score]
                player_stats_list.extend(
                    [stat for stat in re.split(split_pattern, player_stats.text) if stat != "/" and stat != ""])
                player_stats_list.insert(9, agent)
                player_stats_list[8] = team_name if team_name is not None else player_stats_list[8]
                player_stats_list = [stat if stat != " " else np.nan for stat in player_stats_list]
                player_stats_list = [float(stat.replace("%", "")) / 100 if isinstance(stat, str) and "%" in stat else stat for stat in player_stats_list]
                data_table.append(player_stats_list)

        parse_player_stats(team1_player_stats, team1_name)
        parse_player_stats(team2_player_stats, team2_name)

    df = pd.DataFrame(data_table, columns=vlrgg_columns)
    return df


def get_all_vlrgg_url(n_start: int, n_pages: int):
    match_url_list = []
    prefix = "https://www.vlr.gg"
    for i in range(n_start, n_pages):
        match_result_page = f"https://www.vlr.gg/matches/results/?page={i}"
        page = requests.get(match_result_page)
        soup = bs4.BeautifulSoup(page.content, "html.parser")
        match_cards = soup.find_all("div", {"class": "wf-card"})
        for match_card in match_cards:
            for url_element in match_card.find_all("a", {"class": "wf-module-item"}):
                match_url_list.append(prefix + url_element.get("href"))
    return match_url_list


if __name__ == "__main__":
    l1 = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280]
    for i in range(len(l1) - 1):
        url_list = get_all_vlrgg_url(l1[i], l1[i + 1])
        df_list = []
        df_total = pd.DataFrame(columns=vlrgg_columns)
        error_count = 0
        count = 0
        for url in url_list:
            if count % 50 == 0:
                print(count / 50)
            try:
                df = vlrgg_scraper(url)
                if df.shape[0] % 10 == 0:
                    df_list.append(df)
            except:
                error_count += 1
            count += 1
        df_total = pd.concat(df_list, ignore_index=True)
        df_total.to_csv(f"test{l1[i]}-{l1[i + 1]}.csv")
        print(error_count)

