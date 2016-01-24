package main

import (
	"errors"
	"strconv"
	"strings"
)

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
