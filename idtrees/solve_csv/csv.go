package main

import (
	"errors"
	"io/ioutil"
	"strconv"
	"strings"
)

type FieldType int

const (
	Integer FieldType = iota
	String
)

type Field struct {
	Name    string
	Special bool
	Ignore  bool
	Type    FieldType
	Values  []interface{}
}

type Entry struct {
	Values map[*Field]interface{}
}

type CSV struct {
	Fields  []*Field
	Entries []*Entry
}

func ReadCSVFile(file string) (*CSV, error) {
	contents, err := ioutil.ReadFile(file)
	if err != nil {
		return nil, err
	}

	lines := strings.Split(string(contents), "\n")
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}

	if len(lines) == 0 {
		return nil, errors.New("missing header line")
	}

	res := &CSV{Fields: []*Field{}, Entries: []*Entry{}}
	fieldNames := strings.Split(lines[0], ",")
	fieldSeenValues := map[*Field]map[string]bool{}
	for _, fieldName := range fieldNames {
		field := &Field{Name: fieldName, Type: String, Values: []interface{}{}}
		if len(fieldName) > 0 && fieldName[0] == '*' {
			field.Name = field.Name[1:]
			field.Special = true
		} else if len(fieldName) > 0 && fieldName[0] == '_' {
			field.Ignore = true
		}
		res.Fields = append(res.Fields, field)
		fieldSeenValues[field] = map[string]bool{}
	}

	for i := 1; i < len(lines); i++ {
		comps := strings.Split(lines[i], ",")
		if len(comps) != len(fieldNames) {
			return nil, errors.New("row " + strconv.Itoa(i) + " has wrong number of rows")
		}
		entry := &Entry{Values: map[*Field]interface{}{}}
		for j, comp := range comps {
			field := res.Fields[j]
			entry.Values[field] = comp
			if !fieldSeenValues[field][comp] {
				fieldSeenValues[field][comp] = true
				field.Values = append(field.Values, comp)
			}
		}
		res.Entries = append(res.Entries, entry)
	}

	res.processIntegerFields()

	return res, nil
}

func (c *CSV) processIntegerFields() {
FieldLoop:
	for _, field := range c.Fields {
		intValues := []interface{}{}
		for _, value := range field.Values {
			if num, err := strconv.Atoi(value.(string)); err != nil {
				continue FieldLoop
			} else {
				intValues = append(intValues, num)
			}
		}
		field.Type = Integer
		field.Values = intValues
		for _, entry := range c.Entries {
			entry.Values[field], _ = strconv.Atoi(entry.Values[field].(string))
		}
	}
}
