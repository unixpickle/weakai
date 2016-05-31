# Abstract

This is probably the simplest, worst search engine ever. First, it indexes some Wikipedia pages. This process could be greatly improved, since it currently treats every word on a page equally when it should prioritize words near the beginnings of articles. After indexing, it lets you search a query against the index, using Nearest Neighbors to find the pages whose keyword vectors match your query the best.

# Usage

First, you must generate an index. Here is an example of how to do that:

    $ cd nearestneighbors
    $ go run indexer/*.go https://en.wikipedia.org/wiki/Monkey 1000 monkey1000.json
    $ go run prune_index/*.go monkey1000.json monkey1000_pruned.json 0.2

This will generate a file called `monkey1000_pruned.json` with an index of 1000 pages which are somehow relevant to monkeys.

After generating an index, you can search it like so:

    $ go run search/*.go monkey1000_pruned.json "your query here"
