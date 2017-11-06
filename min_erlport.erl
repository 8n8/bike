-module(min_erlport).
-export(testit/0).

testit() ->
    {ok, PythonInstance} = python:start().
