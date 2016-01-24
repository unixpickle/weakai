package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"strings"

	"github.com/unixpickle/weakai/idtrees"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "Usage: idtrees <people.csv>")
		os.Exit(1)
	}

	contents, err := ioutil.ReadFile(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	lines := strings.Split(string(contents), "\n")
	people := []*Person{}
	for i, line := range lines {
		if i == 0 {
			continue
		}
		trimmed := strings.TrimSpace(line)
		if len(trimmed) == 0 {
			continue
		}
		person, err := ParsePerson(trimmed)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		people = append(people, person)
	}

	dataSet := make(idtrees.DataSet, len(people))
	for i, person := range people {
		dataSet[i] = person
	}

	idtrees.CreateBoolField(dataSet, func(p idtrees.Entry) bool {
		return p.(*Person).active
	}, "Is this person still active?")
	idtrees.CreateBoolField(dataSet, func(p idtrees.Entry) bool {
		return p.(*Person).acts
	}, "Does this person act?")
	idtrees.CreateBoolField(dataSet, func(p idtrees.Entry) bool {
		return p.(*Person).sings
	}, "Does this person sing?")
	idtrees.CreateBoolField(dataSet, func(p idtrees.Entry) bool {
		return p.(*Person).female
	}, "Is this person female?")

	idtrees.CreateBisectingIntFields(dataSet, func(p idtrees.Entry) int {
		return p.(*Person).age
	}, "Is the person over %d years old? (Dead people are 0)")
	idtrees.CreateBisectingIntFields(dataSet, func(p idtrees.Entry) int {
		return p.(*Person).children
	}, "Does the person have more than %d child(ren)?")

	treeRoot := idtrees.GenerateTree(dataSet)
	if treeRoot == nil {
		fmt.Fprintln(os.Stderr, "The data is inconclusive.")
		os.Exit(1)
	}

	fmt.Println("Got a tree:")
	fmt.Println(treeRoot)
}
