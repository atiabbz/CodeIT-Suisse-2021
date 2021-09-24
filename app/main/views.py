from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import json
from django.views.decorators.csrf import csrf_exempt


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
