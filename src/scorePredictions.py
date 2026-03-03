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
8.751165,
9.279895,
9.618825,
9.937419,
10.276349,
10.669507,
11.04233,
11.421931,
11.818667,
12.208248,
12.601407,
13.001344,
13.414838,
13.774103,
14.201155,
14.641763,
15.055258,
15.455195,
15.868689,
16.255069,
16.63467,
17.0685,
17.434544
]

other = [
8.745708,
9.481125,
9.628229,
9.840417,
10.647104,
10.862187,
11.036458,
11.210562,
11.421854,
11.803562,
12.195917,
12.594396,
13.006208,
13.227688,
13.393208,
13.771437,
14.186750,
14.630417,
14.879542,
15.262250,
15.447104,
15.853521,
16.234708,
16.616375,
17.030458,
]

r = evaluate_event_predictions(real, other, tolerance=0.05)
print(f"  TP={r['TP']} FP={r['FP']} FN={r['FN']}")
print(f"  Precision={r['precision']:.3f} Recall={r['recall']:.3f}")
print(f"  F1={r['f1']:.3f} average distance between matches={r['avg']:.4f}")
