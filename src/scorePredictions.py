import numpy as np
from scipy.optimize import linear_sum_assignment

import networkx as nx

def evaluate_event_predictions(
    truth, #true list/key of timestamps of catches
    preds, #list of generated guesses on where the catches are
    tolerance=0.05
):
    G = nx.Graph()

    #label nodes for graph
    truth_nodes = [("t", i) for i in range(len(truth))]
    pred_nodes = [("p", j) for j in range(len(preds))]
        #lists of tuples like this:
            #[ (t,0), (t,1),...]
            #signifying truth[0],truth[1],etc
            #[ (p,0), (p,1),...]
            #signifying prediction[0],prediction[1],etc

    #add nodes to graph
    G.add_nodes_from(truth_nodes, bipartite=0) #the 0th parition
    G.add_nodes_from(pred_nodes, bipartite=1) #the 1st partition
    # graph has nodes labled with their bipartite partition
    '''
        {
            (t, 0): {bipartite:0},
            (t, 1): {bipartite:0},
            (p, 0): {bipartite:1},
            (p, 1): {bipartite:1},
            (p, 2): {bipartite:1},
            ...
        }
    '''

    # add edges (matches) where timestamps are within tolerance
    for i, t in enumerate(truth):
        for j, p in enumerate(preds):
            if abs(p - t) <= tolerance:
                G.add_edge(("t", i), ("p", j))
    # G now has something like this:
    '''
        { if this pair is a match
            ("t",1): {("p",0)},
            ("p",0): {("t",1)}
        }
    '''

    # Compute maximum bipartite matching
    matching = nx.algorithms.bipartite.maximum_matching(
        G,
        top_nodes=truth_nodes
    )
    #this now returns a dictionary that has all of the matches,
    # but it has both sides. so it'll have the match of 
    # (truth1,prediction1) and (prediction1,truth1)
    '''matching
    {
        ("t", 0): ("p", 0),
        ("p", 0): ("t", 0),
        ("t", 1): ("p", 1),
        ("p", 1): ("t", 1)
    }
    '''

    # extract the matches / get rid of duplicates
    matches = []
    for node in truth_nodes: #node is a tuple, like (t,1)
        if node in matching:
            t_idx = node[1] #grab just the number
            p_idx = matching[node][1] #grab just the number
                #matching[ (t,1) ][1] = (p,1)[1] = 1
            matches.append((truth[t_idx], preds[p_idx]))
                #matches has:
            ''' truth, predicted
                (1.00, 0.98), 
                (2.01, 2.34),
                ...
            '''

    tp = len(matches)
    fp = len(preds) - tp
    fn = len(truth) - tp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )

    averageDistance = (
        sum(abs(prediction - truth) for truth, prediction in matches) / tp
        if tp > 0 else float("inf")
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg": averageDistance,
        "matches": matches,
    }


# def evaluate_event_predictions(
#     truth, #array of true timestamps
#     preds, #array of generated timestamps
#     tolerance=0.05 #how close a generated timestamp has to be to the truth to be considered a match
# ):
#     i = 0
#     j= 0
#     matches = []

#     # go thru arrays and see if the times match up. 
#     # a prediction can match forwards or backwards
#     while i < len(truth) and j < len(preds):
#         dif = preds[j] - truth[i]

#         #it's a match if the prediction is within the tolerance
#         if abs(dif) <= tolerance:
#             #put the set of truth and its match into a list
#             matches.append((truth[i], preds[j]))
#             i += 1
#             j += 1
#         #if the current prediction is too far behind the truths, 
#         # skip over it and go to next prediction index
#         elif preds[j] < truth[i] - tolerance:
#             j += 1
#         # otherwise increase the truth index
#         else:
#             i += 1

#     tp = len(matches)#how many actual catches we detected 
#     fp = len(preds) - tp#how many of our predictions were wrong
#     fn = len(truth) - tp#how many catches we didnt detect

#     #how many predictions were correct
#     precision = tp / (tp + fp) if (tp + fp) else 0.0

#     #how many true events were found
#     recall = tp / (tp + fn) if (tp + fn) else 0.0

#     #balances and combines precision and recall 
#     f1 = (
#         2 * precision * recall / (precision + recall)
#         if (precision + recall) else 0.0
#     )

#     #see how numerically close each match actually is,
#     #so that we can see when our predictions get closer to exact 
#     #timestamp (like being .04 away vs .001 away)
#     averageDistance = (
#         sum(abs(prediction - truth) for truth, prediction in matches) / tp
#         if tp > 0 else float("inf")
#     )

#     return {
#         "TP": tp,
#         "FP": fp,
#         "FN": fn,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1,
#         "avg": averageDistance, 
#         "matches": matches,
#     }


real = [
6.057569,
6.296212,
6.546788,
6.749635,
7.004188,
7.203058,
7.457611,
7.680345,
7.918988,
8.101948,
8.376388,
8.583213,
8.841743,
9.040613,
9.291189,
9.498013,
9.744611,
9.951436,
10.23383,
10.408836,
10.675321,
10.886123,
11.120789,
11.3475,
11.582166,
11.781036,
12.051499,
12.234459,
12.504922,
12.675949,
12.982209,
13.161191,
13.411767,
13.602682,
13.873144,
14.04815,
14.330545,
14.529414,
14.783967,
14.96295,
15.245345,
15.42035,
15.698768,
15.881728,
16.160145,
16.339128,
16.621523,
16.788573,
17.070968,
17.273815,
17.528368,
17.687464,
17.985768,
18.18066,
18.435213,
18.582377,
18.892614,
19.067619,
19.532974,
]

other = [
6.064563,
6.302458,
6.534104,
6.998250,
7.199708,
7.456042,
7.666750,
7.734042,
7.920292,
8.103729,
8.372729,
8.585646,
8.837083,
9.017437,
9.293479,
9.501729,
9.744458,
9.953938,
10.230688,
10.400729,
10.674812,
10.875521,
11.129458,
11.331854,
11.598458,
11.781083,
12.042292,
12.501625,
12.670979,
13.147104,
13.407937,
13.599875,
13.884438,
14.048042,
14.323521,
14.525854,
14.780479,
14.965375,
15.237104,
15.427938,
15.691812,
15.887042,
16.163896,
16.340625,
16.611146,
16.779542,
17.056750,
17.264458,
17.327833,
17.557604,
17.684729,
17.978458,
18.170458,
18.233521,
18.586417,
18.898688,
19.056021,
19.522229,

]

r = evaluate_event_predictions(real, other, tolerance=0.05)
print(f"  TP={r['TP']} FP={r['FP']} FN={r['FN']}")
print(f"  Precision={r['precision']:.3f} Recall={r['recall']:.3f}")
print(f"  F1={r['f1']:.3f} average distance between matches={r['avg']:.4f}")
