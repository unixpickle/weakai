package main

import (
	"errors"
	"strconv"
	"strings"

	"github.com/unixpickle/weakai/idtrees"
)

type Person struct {
	name           string
	age            int
	children       int
	sings          bool
	acts           bool
	active         bool
	hasBeenMarried bool
	female         bool

	fieldValues map[idtrees.Field]idtrees.Value
}

func ParsePerson(s string) (p *Person, err error) {
	var res Person
	res.fieldValues = map[idtrees.Field]idtrees.Value{}

	comps := strings.Split(s, ",")
	if len(comps) != 8 {
		return nil, errors.New("invalid number of columns")
	}
	res.name = comps[0]
	res.age, err = strconv.Atoi(comps[1])
	if err != nil {
		return
	}
	res.children, err = strconv.Atoi(comps[2])
	if err != nil {
		return
	}
	boolFields := []*bool{&res.sings, &res.acts, &res.active, &res.hasBeenMarried,
		&res.female}
	for i := 3; i < len(comps); i++ {
		if comps[i] == "true" {
			*(boolFields[i-3]) = true
		} else if comps[i] != "false" {
			return nil, errors.New("invalid boolean field: " + comps[i])
		}
	}
	return &res, nil
}

func (p *Person) Class() idtrees.Value {
	if p.hasBeenMarried {
		return idtrees.StringValue("Has been married")
	} else {
		return idtrees.StringValue("Never married")
	}
}

func (p *Person) FieldValues() map[idtrees.Field]idtrees.Value {
	return p.fieldValues
}
