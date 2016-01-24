package main

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"strconv"
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

	// TODO: generate a bunch of questions here.
}

type Person struct {
	Name           string
	Age            int
	Children       int
	Sings          bool
	Acts           bool
	Active         bool
	HasBeenMarried bool
	Female         bool
}

func ParsePerson(s string) (p *Person, err error) {
	var res Person
	comps := strings.Split(s, ",")
	if len(comps) != 8 {
		return nil, errors.New("invalid number of columns")
	}
	res.Name = comps[0]
	res.Age, err = strconv.Atoi(comps[1])
	if err != nil {
		return
	}
	res.Children, err = strconv.Atoi(comps[2])
	if err != nil {
		return
	}
	boolFields := []*bool{&res.Sings, &res.Acts, &res.Active, &res.HasBeenMarried,
		&res.Female}
	for i := 3; i < len(comps); i++ {
		if comps[i] == "true" {
			*(boolFields[i-3]) = true
		} else if comps[i] != "false" {
			return nil, errors.New("invalid boolean field: " + comps[i])
		}
	}
	return &res, nil
}
