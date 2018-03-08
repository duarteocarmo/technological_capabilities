from py2neo import Graph
import numpy as np
from pandas import DataFrame
import itertools
import csv
import seaborn as sns
import json
import math
import pandas as pd
import plotly
import plotly.graph_objs as go
import qgrid
from scipy import stats, spatial
from sklearn.cluster.bicluster import SpectralBiclustering
import operator
from IPython.display import display, HTML


# graph connection
local_connection_url = "http://localhost:7474/db/data"
connection_to_graph = Graph(local_connection_url)


def get_axis():
    """
    Organized dictionnary of all of the terms in the database, categorized.

    """
    query_no_interestions = """     MATCH (a:Asset)-[:CONTAINS]->(fs:Feedstock)
                                    MATCH (a:Asset)-[:CONTAINS]->(out:Output)
                                    MATCH (a:Asset)-[:CONTAINS]->(pt:ProcessingTech)
                                    RETURN fs.term, pt.term, out.term, count(a)
                            """

    query_intersections = """       MATCH (a:Asset)-[:CONTAINS]->(fs:{})
                                    MATCH (a:Asset)-[:CONTAINS]->(t:{})
                                    WHERE fs<>t
                                    RETURN fs.term, t.term, count(a)
                          """

    process_variables = ['Feedstock', 'Output', 'ProcessingTech']
    name_abreviations = ['FS', 'PT', 'OUT']

    # Return query as matrix
    data_no_intersections = DataFrame(connection_to_graph.data(query_no_interestions)).as_matrix()

    # Create a dictionnary with the first names
    names = {'FS': list(set(data_no_intersections[:, 1].tolist())),
             'PT': list(set(data_no_intersections[:, 2].tolist())),
             'OUT': list(set(data_no_intersections[:, 3].tolist()))}

    # Extra labels that only appear in non-intersection queries
    for position, category in enumerate(process_variables):
        data_no_intersections = DataFrame(
            connection_to_graph.data(query_intersections.format(category, category))).as_matrix()
        for column_number in range(1, 3):
            column = data_no_intersections[:, column_number]
            for name in column:
                if name not in names[name_abreviations[position]]:
                    names[name_abreviations[position]].append(name)

    return names


def get_country_names():
    """
    Returns a list with all of the countries in the database.

    """
    country_query = """ MATCH (n:Country) 
                        WITH n.name AS Country 
                        RETURN Country;
                        """

    country_names = list(set(DataFrame(connection_to_graph.data(country_query)).as_matrix()[:, 0]))

    return country_names


def get_country_matrix(country, normalization=True):
    """
    Gets a country capability matrix.

    """

    names = get_axis()
    matrix_axis_names = []

    for category in names:
        matrix_axis_names += names[category]

    # define queries
    country_no_interestions = """   MATCH (a:Asset)-[:CONTAINS]->(fs:Feedstock)
                                    MATCH (a:Asset)-[:CONTAINS]->(out:Output)
                                    MATCH (a:Asset)-[:CONTAINS]->(pt:ProcessingTech)
                                    WHERE a.country = "{}"
                                    RETURN fs.term, pt.term, out.term, count(a)
                                    """.format(country)

    process_variables = ['Feedstock', 'Output', 'ProcessingTech']

    country_intersections = """     MATCH (a:Asset)-[:CONTAINS]->(fs:{})
                                    MATCH (a:Asset)-[:CONTAINS]->(t:{})
                                    WHERE fs<>t AND a.country = "{}"
                                    RETURN fs.term, t.term, count(a)
                                    """
    # get data
    data_no_intersections = DataFrame(connection_to_graph.data(country_no_interestions)).as_matrix()

    # create matrix
    country_matrix = np.zeros([len(matrix_axis_names), len(matrix_axis_names)])

    # for no intersections data
    for row in data_no_intersections:
        # the last column is the frequency (count)
        frequency = row[0]
        indexes = [matrix_axis_names.index(element) for element in row[1::]]
        # add frequency value to matrix position
        for pair in itertools.combinations(indexes, 2):
            country_matrix[pair[0], pair[1]] += frequency
            country_matrix[pair[1], pair[0]] += frequency

    # for intersecting data
    for category in process_variables:
        process_data = DataFrame(
            connection_to_graph.data(country_intersections.format(category, category, country))).as_matrix()
        for row in process_data:
            frequency = row[0]
            indexes = [matrix_axis_names.index(element) for element in row[1::]]
            # add frequency value to matrix position
            for pair in itertools.combinations(indexes, 2):
                country_matrix[pair[0], pair[1]] += frequency / 2  # Divided by two because query not optimized
                country_matrix[pair[1], pair[0]] += frequency / 2  # Divided by two because query not optimized

    # normalize
    normalized_country_matrix = (country_matrix - np.mean(country_matrix)) / np.std(country_matrix)

    # dynamic return
    if normalization is True:
        return normalized_country_matrix
    else:
        return country_matrix


def get_list_from(matrix):
    """
    Transforms capability matrix into list.

    """

    only_valuable = []
    counter = 1
    for row_number in range(matrix.shape[0]):
        only_valuable += matrix[row_number, counter::].tolist()
        counter += 1

    return only_valuable


def export_data_to_csv():
    """
    Exports data to csv.
    """

    with open('country_data.csv', 'wb') as f:
        writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_NONE)
        for country in get_country_names():
            data = get_list_from(get_country_matrix(country, normalization=False))
            writer.writerow(data)


def export_names_to_csv():
    """
    Exports data, each list to a csv line.
    """

    with open('country_names.csv', 'wb') as f:

        writer = csv.writer(f, delimiter=',')
        for country in get_country_names():
            writer.writerow([country])


export_data_to_csv()














