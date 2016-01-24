package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"sort"
	"strings"
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

	dataSet := make(DataSet, len(people))
	for i, person := range people {
		dataSet[i] = &PersonEntry{person, map[*Question]string{}}
	}

	addBoolQuestion(dataSet, func(p *Person) bool {
		return p.Active
	}).Prompt = "Is this person still active?"
	addBoolQuestion(dataSet, func(p *Person) bool {
		return p.Acts
	}).Prompt = "Does this person act?"
	addBoolQuestion(dataSet, func(p *Person) bool {
		return p.Sings
	}).Prompt = "Does this person sing?"
	addBoolQuestion(dataSet, func(p *Person) bool {
		return p.HasBeenMarried
	}).Prompt = "Has this person been married?"

	addIntQuestion(dataSet, func(p *Person) int {
		return p.Age
	}, "Is the person over %d years old? (Dead people are 0)")
	addIntQuestion(dataSet, func(p *Person) int {
		return p.Children
	}, "Does the person have more than %d child(ren)?")

	treeRoot := GenerateIDTree(dataSet)
	if treeRoot == nil {
		fmt.Fprintln(os.Stderr, "The data is inconclusive.")
		os.Exit(1)
	}

	fmt.Println("Got a useful tree:")
	fmt.Println(treeRoot)
}

type PersonEntry struct {
	person      *Person
	questionMap map[*Question]string
}

func (p *PersonEntry) Class() int {
	if p.person.Female {
		return 1
	} else {
		return 0
	}
}

func (p *PersonEntry) QuestionAnswers() map[*Question]string {
	return p.questionMap
}

func addBoolQuestion(entries DataSet, getter func(p *Person) bool) *Question {
	q := &Question{Answers: []string{"yes", "no"}}
	for _, entry := range entries {
		personEntry := entry.(*PersonEntry)
		answer := "no"
		if getter(personEntry.person) {
			answer = "yes"
		}
		personEntry.questionMap[q] = answer
	}
	return q
}

func addIntQuestion(entries DataSet, getter func(p *Person) int, prompt string) {
	possibilities := []int{}
	seenPossibilities := map[int]bool{}
	for _, entry := range entries {
		val := getter(entry.(*PersonEntry).person)
		if !seenPossibilities[val] {
			seenPossibilities[val] = true
			possibilities = append(possibilities, val)
		}
	}
	sort.Ints(possibilities)

	for i := 0; i < len(possibilities)-1; i++ {
		middle := (possibilities[i] + possibilities[i+1]) / 2
		question := &Question{
			Prompt:  fmt.Sprintf(prompt, middle),
			Answers: []string{"yes", "no"},
		}
		for _, entry := range entries {
			personEntry := entry.(*PersonEntry)
			if getter(personEntry.person) > middle {
				personEntry.questionMap[question] = "yes"
			} else {
				personEntry.questionMap[question] = "no"
			}
		}
	}
}
