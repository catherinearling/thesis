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


real = [
6.16667,
6.527498,
6.910897,
7.250618,
7.643724,
7.988299,
8.366845,
8.687153,
9.094819,
9.439393,
9.827646,
10.133395,
10.473116,
10.953578,
11.157411,
11.492279,
11.846559,
12.20084,
12.550267,
12.904548,
13.253975,
13.593696,
13.96739,
14.32167,
14.685657,
15.025378,
15.365099,
15.714527,
16.112486,
16.405333,
16.821047,
17.175327,
17.495636,
17.888741,
18.218756,
18.597303,
18.937024,
19.247626,
19.60676,
19.980453,
20.334733,
20.664748,
20.999616,
21.35875,
21.732443,
22.077017,
22.407032,
]

other = [
6.210854,
6.527396,
6.908083,
7.241042,
7.629937,
7.965292,
8.355937,
8.682875,
8.791625,
9.078375,
9.427854,
9.805250,
10.129625,
10.455854,
10.945667,
11.151729,
11.478958,
11.841167,
12.193833,
12.566062,
12.900333,
13.260896,
13.595771,
13.667438,
13.947313,
14.317750,
14.680229,
15.032354,
15.095854,
15.362354,
16.102563,
16.400562,
16.815396,
17.160479,
17.237521,
17.481667,
17.864583,
18.211521,
18.586854,
18.925667,
19.235208,
19.353813,
19.984333,
20.336833,
20.662417,
21.341833,
21.710375,
22.405604,

]

r = evaluate_event_predictions(real, other, tolerance=0.05)
print(f"  TP={r['TP']} FP={r['FP']} FN={r['FN']}")
print(f"  Precision={r['precision']:.3f} Recall={r['recall']:.3f}")
print(f"  F1={r['f1']:.3f} average distance between matches={r['avg']:.4f}")
