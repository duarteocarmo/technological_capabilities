from py2neo import Graph
from pandas import DataFrame

graph_url = "http://localhost:7474/db/data"
query = """ MATCH p1=(feed:Feedstock)<-[:CONTAINS]-(a1:Asset)-[:CONTAINS]->(proc:ProcessingTech)
            MATCH p2=(proc:ProcessingTech)<-[:CONTAINS]-(a1:Asset)-[:CONTAINS]->(out:Output) 
            WITH feed.term AS Feedstock, proc.term AS Processing_Technology, out.term AS Output, count(p1) AS count 
            RETURN Feedstock, Processing_Technology, Output, count 
            ORDER BY count 
            DESC LIMIT 300"""
graph = Graph(graph_url)

# get data as a dump
# graph.run(query).dump()

# get data as dict
# output = graph.data(query)

# get data as Pd
a = DataFrame(graph.data(query))
print a

graph.run("CALL db.schema()").dump()
