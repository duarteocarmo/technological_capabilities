from py2neo import Graph
import numpy as np 
from pandas import DataFrame
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

# Create connection to Neo4j
local_connection_url = "http://localhost:7474/db/data"
connection_to_graph = Graph(local_connection_url)

# Queries
limit = 200
process_variables = ['Feedstock', 'Output', 'ProcessingTech']

query_no_interestions = """     MATCH (a:Asset)-[:CONTAINS]->(fs:Feedstock)
                                MATCH (a:Asset)-[:CONTAINS]->(out:Output)
                                MATCH (a:Asset)-[:CONTAINS]->(pt:ProcessingTech)
                                RETURN fs.term, pt.term, out.term, count(a)
                        """

# issue: this query needs to be divided by two when building the matrix -> NON OPTIMIZED
query_intersections = """       MATCH (a:Asset)-[:CONTAINS]->(fs:{})
                                MATCH (a:Asset)-[:CONTAINS]->(t:{})
                                WHERE fs<>t
                                RETURN fs.term, t.term, count(a)
                      """


# Return dataframes
data_no_intersections = DataFrame(connection_to_graph.data(query_no_interestions)).as_matrix()



# Axis Names
feedstock_names = set(list(data_no_intersections[:, 1]))
processing_technology_names = set(list(data_no_intersections[:, 2]))
output_names = set(list(data_no_intersections[:, 3]))
matrix_axis_names = list(feedstock_names) + list(processing_technology_names) + list(output_names)

# Build Matrix
# initialize matrix
matrix = np.zeros([len(matrix_axis_names), len(matrix_axis_names)])


# find index of element in a given list
def find_index(something, in_list):
    return in_list.index(something)


# for every row in original response
for row in data_no_intersections:
    # the last column is the frequency
    frequency = row[0]
    indexes = [find_index(element, matrix_axis_names) for element in row[1::]]
#     add frequency value to matrix position
    for pair in itertools.combinations(indexes, 2):
        matrix[pair[0], pair[1]] += frequency
        matrix[pair[1], pair[0]] += frequency

not_found = []

for category in process_variables:
    print 'Processing ', category
    process_data = DataFrame(connection_to_graph.data(query_intersections.format(category, category))).as_matrix()

    for row in process_data:
        stop = 0
        for element in row[1::]:
            if element not in matrix_axis_names:
                not_found.append(element)
                stop = 1
        if stop != 1:
            frequency = row[0]
            indexes = [find_index(element, matrix_axis_names) for element in row[1::]]
            #     add frequency value to matrix position
            for pair in itertools.combinations(indexes, 2):
                matrix[pair[0], pair[1]] += frequency / 2
                matrix[pair[1], pair[0]] += frequency / 2



# normalize matrix using scipy normalization available here:
# http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-normalization
normalized_matrix = (matrix - np.mean(matrix)) / np.std(matrix)


def check_symmetric(a, tol):
    return np.allclose(a, a.T, atol=tol)

print check_symmetric(matrix, 1e-8)
print check_symmetric(normalized_matrix, 1e-8)
for i in range(200):
    print normalized_matrix[i, i], matrix[i, i]