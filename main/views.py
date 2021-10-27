from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, request
import json
from django.views.decorators.csrf import csrf_exempt
import numpy as np
from scipy.stats import norm
import sys
from fractions import Fraction


def index(request):
    return render(request, "main/index.html")


def stringTest(request):
    response = "foo"
    return HttpResponse(response)


def jsonTest(request):
    response = {"foo": "bar"}
    return JsonResponse(response)


@csrf_exempt
def consumeJson(request):
    data = json.loads(request.body)
    response = {"received": data}
    return JsonResponse(response)


@csrf_exempt
def asteroid(request):
    body = json.loads(request.body)
    test_cases = body["test_cases"]
    response = []

    for case in test_cases:
        collapse = []
        s = 0
        for i in range(1, len(case)):
            if case[i] != case[i - 1]:
                collapse.append((case[i - 1], (s, i - 1)))
                s = i
        collapse.append((case[-1], (s, len(case) - 1)))
        print(collapse)
        best = np.NINF
        origin = None

        for i in range(len(collapse)):
            lp = i
            rp = i
            score = 0

            while lp >= 0 and rp < len(collapse) and collapse[lp][0] == collapse[rp][0]:
                u = collapse[lp]
                v = collapse[rp]

                ll = u[1][1] - u[1][0] + 1
                rl = v[1][1] - v[1][0] + 1

                total = ll
                if lp != rp:
                    total = ll + rl
                if total >= 10:
                    score += total * 2
                elif total >= 7:
                    score += total * 1.5
                else:
                    score += total

                lp -= 1
                rp += 1

            if score > best:
                best = score
                origin = (collapse[i][1][0] + collapse[i][1][1]) // 2

        response.append({"input": case, "score": best, "origin": origin})

    return JsonResponse(response, safe=False)


def expected_return_per_view(option_dict, view_dict):

    strike = option_dict["strike"]
    premium = option_dict["premium"]

    if option_dict["type"] == "call":

        if strike >= view_dict["max"]:
            return -premium
        else:
            a = view_dict["min"]
            b = view_dict["max"]
            c = max(a, strike)
            mu = view_dict["mean"]
            sigma = view_dict["var"]
            alpha = (a - mu) / sigma
            beta = (b - mu) / sigma
            gamma = (c - mu) / sigma

            multiplier1 = (norm.cdf(beta) - norm.cdf(gamma)) / (
                norm.cdf(beta) - norm.cdf(alpha)
            )
            multiplier2 = (
                mu
                - strike
                - (
                    sigma
                    * sigma
                    * (norm.pdf(beta) - norm.pdf(gamma))
                    / (norm.cdf(beta) - norm.cdf(gamma))
                )
            )
            return (multiplier1 * multiplier2) - premium

    else:

        if strike <= view_dict["min"]:
            return -premium
        else:
            a = view_dict["min"]
            b = view_dict["max"]
            c = min(b, strike)
            mu = view_dict["mean"]
            sigma = view_dict["var"]
            alpha = (a - mu) / sigma
            beta = (b - mu) / sigma
            gamma = (c - mu) / sigma

            multiplier1 = (norm.cdf(gamma) - norm.cdf(alpha)) / (
                norm.cdf(beta) - norm.cdf(alpha)
            )
            multiplier2 = (
                strike
                - mu
                + (
                    sigma
                    * sigma
                    * (norm.pdf(gamma) - norm.pdf(alpha))
                    / (norm.cdf(gamma) - norm.cdf(alpha))
                )
            )
            return (multiplier1 * multiplier2) - premium


def expected_return_all_views(option_dict, view_dicts):
    numerator = 0
    denominator = 0

    for view_dict in view_dicts:
        numerator += view_dict["weight"] * expected_return_per_view(
            option_dict, view_dict
        )
        denominator += view_dict["weight"]

    return numerator / denominator


@csrf_exempt
def evaluateOptions(request):
    data = json.loads(request.body)
    # logging.info("data sent for evaluation {}".format(data))

    max_val = 0
    max_abs_val = 0
    max_pos = 0
    for pos in range(len(data["options"])):
        val = expected_return_all_views(data["options"][pos], data["view"])
        abs_val = abs(val)
        if max_abs_val < abs_val:
            max_abs_val = abs_val
            max_val = val
            max_pos = pos

    result = [0] * len(data["options"])
    if max_val < 0:
        result[max_pos] = -100
    else:
        result[max_pos] = 100

    # logging.info("My result :{}".format(result))
    return JsonResponse(result, safe=False)


@csrf_exempt
def evaluateParasite(request):
    data = json.loads(request.body)
    # logging.info("data sent for evaluation {}".format(data))
    output = []
    for testcase in data:
        testcase_output = parasite(testcase)
        output.append(testcase_output)

    # logging.info("My result :{}".format(output))
    return JsonResponse(output, safe=False)


def parasite(info):
    results = {
        "room": info["room"],
        "p1": {indiv: -1 for indiv in info["interestedIndividuals"]},
        "p2": 0,
        "p3": 0,
        "p4": 0,
    }
    # Part 1
    grid_status_p1 = [x[:] for x in info["grid"]]
    r, c = len(grid_status_p1), len(grid_status_p1[0])
    indiv_coord = []
    for indiv in info["interestedIndividuals"]:
        coords = tuple(map(int, indiv.split(",")))
        if grid_status_p1[coords[0]][coords[1]] == 0:
            results["p1"][indiv] = -1
        elif grid_status_p1[coords[0]][coords[1]] == 1:
            no_one_around = all(
                [
                    grid_status_p1[n[0]][n[1]] == 0
                    for n in find_neighbors(coords, grid_status_p1)
                ]
            )
            if no_one_around:
                results["p1"][indiv] = -1
                continue
            else:
                indiv_coord.append(coords)
        elif grid_status_p1[coords[0]][coords[1]] == 2:
            results["p1"][indiv] = -1
        elif grid_status_p1[coords[0]][coords[1]] == 3:
            results["p1"][indiv] = 0

    time = 0
    current_state_p1 = [x[:] for x in grid_status_p1]
    prev_grid_status_p1 = [x[:] for x in grid_status_p1]
    while True:
        time += 1
        for i in range(r):
            for j in range(c):
                if prev_grid_status_p1[i][j] == 1:
                    nn = find_neighbors((i, j), grid_status_p1)
                    infected_nearby = any(
                        [prev_grid_status_p1[n[0]][n[1]] == 3 for n in nn]
                    )
                    if infected_nearby:
                        current_state_p1[i][j] = 3
                        if (i, j) in indiv_coord:
                            results["p1"][str(i) + "," + str(j)] = time
        if prev_grid_status_p1 == current_state_p1:
            break
        prev_grid_status_p1 = [x[:] for x in current_state_p1]
    # Part 2
    healthy_remains = any(
        [prev_grid_status_p1[i][j] == 1 for j in range(c) for i in range(r)]
    )
    if healthy_remains:
        results["p2"] = -1
    else:
        results["p2"] = time - 1
    # Part 3
    time = 0
    current_state_p3 = [x[:] for x in grid_status_p1]
    prev_grid_status_p3 = [x[:] for x in grid_status_p1]
    while True:

        time += 1
        for i in range(r):
            for j in range(c):
                if prev_grid_status_p3[i][j] == 1:
                    nn = find_neighbors_mutated((i, j), grid_status_p1)
                    infected_nearby = any(
                        [prev_grid_status_p3[n[0]][n[1]] == 3 for n in nn]
                    )
                    if infected_nearby:
                        current_state_p3[i][j] = 3
        if prev_grid_status_p3 == current_state_p3:
            break
        prev_grid_status_p3 = [x[:] for x in current_state_p3]
    healthy_remains_p3 = any(
        [prev_grid_status_p3[i][j] == 1 for j in range(c) for i in range(r)]
    )
    if healthy_remains_p3:
        results["p3"] = -1
    else:
        results["p3"] = time - 1
    return results


def find_neighbors(coords, grid):
    r, c = len(grid), len(grid[0])
    directions = [[0, 1], [1, 0], [-1, 0], [0, -1]]
    neighbors = []
    for d in directions:
        if (
            (0 <= coords[0] + d[0])
            and (coords[0] + d[0] < r)
            and (0 <= coords[1] + d[1])
            and (coords[1] + d[1] < c)
        ):
            neighbors.append((coords[0] + d[0], coords[1] + d[1]))
    return neighbors


def find_neighbors_mutated(coords, grid):
    r, c = len(grid), len(grid[0])
    directions = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]]
    neighbors = []
    for d in directions:
        if (
            (0 <= coords[0] + d[0])
            and (coords[0] + d[0] < r)
            and (0 <= coords[1] + d[1])
            and (coords[1] + d[1] < c)
        ):
            neighbors.append((coords[0] + d[0], coords[1] + d[1]))
    return neighbors


@csrf_exempt
def evaluateStockHunter(request):
    data = json.loads(request.body)
    # logging.info("data sent for evaluation {}".format(data))
    output = []
    for i in data:
        entry_point = i["entryPoint"]
        target_point = i["targetPoint"]
        gridDepth = i["gridDepth"]
        gridKey = i["gridKey"]
        horizontalStepper = i["horizontalStepper"]
        verticalStepper = i["verticalStepper"]
        soln = stockhunter(
            entry_point,
            target_point,
            gridDepth,
            gridKey,
            horizontalStepper,
            verticalStepper,
        )
        output.append(soln)

    # logging.info("My result :{}".format(output))
    return JsonResponse(output, safe=False)


import dijkstra


def find_risk_index(x, y, h, v, t):
    if (x == 0 and y == 0) or (x == t["first"] and y == t["second"]):
        return 0
    elif y == 0:
        return x * h
    elif x == 0:
        return y * v
    else:
        return find_risk_index(x - 1, y, h, v, t) * find_risk_index(x, y - 1, h, v, t)


def find_neighbor(x, y, c, r):
    nn = []
    direction = [[0, 1], [1, 0], [-1, 0], [0, -1]]
    for d in direction:
        if (0 <= x + d[0] < c) and (0 <= y + d[1] < r):
            nn.append((x + d[0], y + d[1]))
    return nn


def stockhunter(
    entry_point, target_point, gridDepth, gridKey, horizontalStepper, verticalStepper
):
    output = {}
    c, r = target_point["first"] + 10, target_point["second"] + 10
    risk_index = [[0 for i in range(c)] for j in range(r)]
    risk_level = [[0 for i in range(c)] for j in range(r)]
    risk_cost = [[0 for i in range(c)] for j in range(r)]
    grid = [[0 for i in range(c)] for j in range(r)]
    for x in range(c):
        for y in range(r):
            if (x == 0 and y == 0) or (x == c - 1 and y == r - 1):
                risk_index[y][x] = 0
                risk_level[y][x] = (risk_index[y][x] + gridDepth) % gridKey
                risk_cost[y][x] = 3 - risk_level[y][x] % 3
            elif x == 0:
                risk_index[y][x] = y * verticalStepper
                risk_level[y][x] = (risk_index[y][x] + gridDepth) % gridKey
                risk_cost[y][x] = 3 - risk_level[y][x] % 3
            elif y == 0:
                risk_index[y][x] = x * horizontalStepper
                risk_level[y][x] = (risk_index[y][x] + gridDepth) % gridKey
                risk_cost[y][x] = 3 - risk_level[y][x] % 3
            else:
                risk_index[y][x] = risk_level[y - 1][x] * risk_level[y][x - 1]
                risk_level[y][x] = (risk_index[y][x] + gridDepth) % gridKey
                risk_cost[y][x] = 3 - risk_level[y][x] % 3
    label = {1: "S", 2: "M", 3: "L"}
    grid = [
        [
            label[risk_cost[y][x]]
            for x in range(entry_point["first"], target_point["first"] + 1)
        ]
        for y in range(entry_point["second"], target_point["second"] + 1)
    ]
    graph = dijkstra.Graph()
    for y in range(len(risk_cost)):
        for x in range(len(risk_cost[0])):
            for n in find_neighbor(x, y, c, r):
                graph.add_edge((x, y), n, risk_cost[n[0]][n[1]])
                print((x, y), n, risk_cost[n[0]][n[1]])
    solve = dijkstra.DijkstraSPF(graph, (entry_point["first"], entry_point["second"]))
    output["gridMap"] = grid
    output["minimumCost"] = solve.get_distance(
        (target_point["first"], target_point["second"])
    )
    return output


import hashlib
from math import log


@csrf_exempt
def cipher_cracking(request):
    input = json.loads(request.body)
    # logging.info("Input: {}".format(input))

    output = []
    for test_case in input:
        X = int(test_case["X"])
        fx = (X + 1) / X * (0.57721566 + log(X) + 0.5 / X) - 1
        FX = "::{:.3f}".format(fx)
        for K in range(1, 10 ** test_case["D"]):
            if (
                hashlib.sha256((str(K) + FX).encode("utf-8")).hexdigest()
                == test_case["Y"]
            ):
                break
        output.append(K)

    # logging.info("Output: {}".format(output))
    return JsonResponse(output, safe=False)


@csrf_exempt
def stockHunting(request):
    stocks = json.loads(request.body)
    result = []
    for s in stocks:
        output = processStocks(s)
        result.append(output)
    return JsonResponse(result, safe=False)


def processStocks(s):
    eP = s["entryPoint"]
    tP = s["targetPoint"]
    x, y = eP["first"], eP["second"]
    xT, yT = tP["first"], tP["second"]
    gridKey = s["gridKey"]
    gridDepth = s["gridDepth"]
    hStep = s["horizontalStepper"]
    vStep = s["verticalStepper"]

    rows = abs(yT - y) + 1
    cols = abs(xT - x) + 1
    gridMap = [[0] * (cols) for _ in range(rows)]
    value_gridMap = [[0] * (cols) for _ in range(rows)]
    # Compute the cost and assign gridMap
    computedValues = [[0] * (cols) for _ in range(rows)]
    for y in range(rows):
        for x in range(cols):
            riskLevel = computeRiskLevel(
                x, y, hStep, vStep, gridKey, gridDepth, computedValues
            )
            # print("Risklevel for x y ", x,y,": ", riskLevel)
            if riskLevel % 3 == 0:
                gridMap[y][x] = "L"
                value_gridMap[y][x] = 3
            elif riskLevel % 3 == 1:
                gridMap[y][x] = "M"
                value_gridMap[y][x] = 2
            elif riskLevel % 3 == 2:
                gridMap[y][x] = "S"
                value_gridMap[y][x] = 1

    # print(gridMap)
    output = {}
    output["gridMap"] = gridMap
    output["minimumCost"] = minCost(value_gridMap, xT, yT)
    return output


def computeRiskLevel(x, y, hStep, vStep, gridKey, gridDepth, computedValues):
    if computedValues[x][y] != 0:
        return computedValues[x][y]
    if x == 0 and y == 0:
        computedValues[x][y] = gridDepth % gridKey
    elif x == 0:
        computedValues[x][y] = (y * vStep + gridDepth) % gridKey
    elif y == 0:
        computedValues[x][y] = (x * hStep + gridDepth) % gridKey
    else:
        computedValues[x][y] = (
            computeRiskLevel(x - 1, y, hStep, vStep, gridKey, gridDepth, computedValues)
            * computeRiskLevel(
                x, y - 1, hStep, vStep, gridKey, gridDepth, computedValues
            )
            + gridDepth
        ) % gridKey

    return computedValues[x][y]


# Returns cost of minimum cost path from (0,0) to (m, n) in mat[R][C]
def minCost(cost, m, n):
    if n < 0 or m < 0:
        return sys.maxsize
    elif m == 0 and n == 0:
        return cost[m][n]
    else:
        return cost[m][n] + min(
            minCost(cost, m - 1, n - 1),
            minCost(cost, m - 1, n),
            minCost(cost, m, n - 1),
        )


def min(x, y, z):
    if x < y:
        return x if (x < z) else z
    else:
        return y if (y < z) else z


from scipy.stats import truncnorm
import math


@csrf_exempt
def calculate(request):
    data = json.loads(request.body)
    options, gauss = data["options"], data["view"]
    result = solve(options, gauss)
    return JsonResponse(result, safe=False)


def solve(options, gauss):
    gaussians = [
        truncnorm(
            (i["min"] - i["mean"]) / math.sqrt(i["var"]),
            (i["max"] - i["mean"]) / math.sqrt(i["var"]),
            loc=i["mean"],
            scale=math.sqrt(i["var"]),
        )
        for i in gauss
    ]
    rv = gaussians[0]
    x = np.linspace(rv.ppf(0.0001), rv.ppf(0.9999), 1000)
    returns = [
        np.sum(
            np.multiply(
                rv.pdf(x),
                np.where(
                    x < j["strike"], -j["premium"], x - (j["strike"] + j["premium"])
                ),
            )
        )
        if j["type"] == "call"
        else np.sum(
            np.multiply(
                rv.pdf(x),
                np.where(
                    x > j["strike"], -j["premium"], (j["strike"] - j["premium"]) - x
                ),
            )
        )
        for j in options
    ]
    ans = [0] * len(returns)
    if max(returns) >= abs(min(returns)):
        ans[returns.index(max(returns))] = 100
    else:
        ans[returns.index(min(returns))] = -100
    return ans

from fractions import Fraction

@csrf_exempt
def evaluateInterviews(request):
    interviews = json.loads(request.body)
    result = []
    for i in interviews:
        result.append(processInterview(i))
    return JsonResponse(result, safe=False)

def processInterview(i):
    questions = i["questions"]
    MAX = i["maxRating"]
    from_list = []
    to_list = []
    total_range = 0
    for q in questions:
        # Traverse through each question's set of ranges
        for i in q:
            to_list.append(i["to"])
            from_list.append(i["from"])
    mergeSort(from_list)
    mergeSort(to_list)
    total_range = to_list[-1] - from_list[0] + 1
    print(total_range)
    probability = Fraction(total_range,MAX)
    p = probability.numerator
    q = probability.denominator
    output = {}
    output["p"], output["q"] = p,q
    print(output)
    return output

def mergeSort(arr):
    if len(arr) > 1:
         # Finding the mid of the array
        mid = len(arr)//2
        # Dividing the array elements
        L = arr[:mid]

        # into 2 halves
        R = arr[mid:]

        # Sorting the first half
        mergeSort(L)

        # Sorting the second half
        mergeSort(R)

        i = j = k = 0

        # Copy data to temp arrays L[] and R[]
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        # Checking if any element was left
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
