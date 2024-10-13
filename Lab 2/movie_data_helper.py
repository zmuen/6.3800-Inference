"""
movie_data_helper.py
Original author: Felix Sun (6.008 TA, Fall 2015)
Modified by:
- Danielle Pace (6.008 TA, Fall 2016),
- George H. Chen (6.008/6.008.1x instructor, Fall 2016),
- Junshen Xu (6.3800 TA, Fall 2022)
- Chirag Falor (6.380 TA, Fall 2024)

***Do not modify this file.***

This file has a number of helper files for the movie rating problem, to
interact with the data files and return relevant information.
"""

import numpy as np
from os.path import isfile
from sys import exit

data_dir = "data" # Modify this if you want to change the folder name

def _check_movie_exists(movie_id):
    """
    Checks if a movie exists.

    Input
    -----
    - movie_id: integer from 0 to <number of movies - 1> specifying which movie

    Output
    ------
    - None
    """

    filename_to_check = f"{data_dir}/ratingsMovie{movie_id}.dat"
    if not (isfile(filename_to_check)):
        exit(f"Movie ID {movie_id} does not exist")


def get_movie_id_list():
    """
    This function returns a 1D NumPy array of all movie ID's based on data in
    the data folder.

    Output
    ------
    1D NumPy array of all the movie ID's
    """

    return np.loadtxt(
        f"{data_dir}/movieNames.dat",
        dtype="int32",
        delimiter="\t",
        usecols=(0,),
        encoding="latin1",
    )


def get_movie_name(movie_id):
    """
    Gets a movie name given a movie ID.

    Input
    -----
    - movie_id: integer from 0 to <number of movies - 1> specifying which movie

    Output
    ------
    - movie_name: string containing the movie name
    """

    # -------------------------------------------------------------------------
    # ERROR CHECK
    #
    _check_movie_exists(movie_id)
    #
    # END OF ERROR CHECK
    # -------------------------------------------------------------------------

    movies = np.loadtxt(
        f"{data_dir}/movieNames.dat",
        dtype={"names": ("movieid", "moviename"), "formats": ("int32", "U100")},
        delimiter="\t",
        encoding="latin1",
    )
    movie_name = movies[movies["movieid"] == movie_id]["moviename"][0].lstrip()
    return movie_name


def get_ratings(movie_id):
    """
    Gets all the ratings for a given movie.

    Input
    -----
    - movie_id: integer from 0 to <number of movies - 1> specifying which movie

    Output
    ------
    - ratings: 1D array consisting of the ratings for the given movie
    """
    # -------------------------------------------------------------------------
    # ERROR CHECK
    #
    _check_movie_exists(movie_id)
    #
    # END OF ERROR CHECK
    # -------------------------------------------------------------------------

    data = np.loadtxt(
        f"{data_dir}/ratingsMovie{movie_id}.dat",
        dtype={"names": ("userid", "rating"), "formats": ("int32", "int32")},
        delimiter="\t",
    )
    ratings = data["rating"]
    return ratings
