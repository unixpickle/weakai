package main

import (
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/unixpickle/weakai/idtrees"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "Usage: idtrees <data.csv>")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "  The first row of the input CSV file specifies field names.")
		fmt.Fprintln(os.Stderr, "  Fields with names starting with _ are ignored.")
		fmt.Fprintln(os.Stderr, "  The field whose name begins with * is identified by the tree.")
		fmt.Fprintln(os.Stderr, "")
		os.Exit(1)
	}

	log.Println("Reading CSV file...")

	csv, err := ReadCSVFile(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	numSpecial := 0
	for _, field := range csv.Fields {
		if field.Special {
			numSpecial++
		}
	}
	if numSpecial != 1 {
		fmt.Fprintln(os.Stderr, "One field's name must start with *, indicating that it is "+
			"the field to identify.")
		os.Exit(1)
	}

	log.Println("Generating data set entries...")
	dataSet := generateFieldlessDataSet(csv)
	log.Println("Generating fields...")
	generateFields(csv, dataSet)

	log.Println("Generating tree...")
	treeRoot := idtrees.GenerateTree(dataSet)
	if treeRoot == nil {
		fmt.Fprintln(os.Stderr, "The data is inconclusive.")
		os.Exit(1)
	}

	log.Println("Printing out the tree...")

	fmt.Println(treeRoot)
}

func generateFieldlessDataSet(csv *CSV) *idtrees.DataSet {
	var specialField *Field
	for _, field := range csv.Fields {
		if field.Special {
			specialField = field
			break
		}
	}

	dataSet := &idtrees.DataSet{
		Entries: make([]idtrees.Entry, len(csv.Entries)),
		Fields:  []idtrees.Field{},
	}

	for i, entry := range csv.Entries {
		var class idtrees.Value
		v := entry.Values[specialField]
		if specialField.Type == Integer {
			class = idtrees.StringValue(specialField.Name + " = " + strconv.Itoa(v.(int)))
		} else {
			class = idtrees.StringValue(specialField.Name + " = " + v.(string))
		}
		dataSet.Entries[i] = &TreeEntry{
			csvEntry: entry,
			values:   []idtrees.Value{},
			class:    class,
		}
	}

	return dataSet
}

func generateFields(csv *CSV, dataSet *idtrees.DataSet) {
	for _, field := range csv.Fields {
		if field.Ignore || field.Special {
			continue
		}
		addField(dataSet, field)
	}
}

func addField(dataSet *idtrees.DataSet, field *Field) {
	adder := func(e idtrees.Entry, v idtrees.Value) {
		treeEntry := e.(*TreeEntry)
		treeEntry.values = append(treeEntry.values, v)
	}
	switch field.Type {
	case Integer:
		idtrees.CreateBisectingIntFields(dataSet, func(e idtrees.Entry) int {
			csvEntry := e.(*TreeEntry).csvEntry
			return csvEntry.Values[field].(int)
		}, adder, field.Name+" > %d")
	case String:
		vals := map[string]*StringPointerValue{}
		idtrees.CreateListField(dataSet, func(e idtrees.Entry) idtrees.Value {
			csvEntry := e.(*TreeEntry).csvEntry
			s := csvEntry.Values[field].(string)
			if ptr, ok := vals[s]; ok {
				return ptr
			} else {
				x := StringPointerValue(s)
				vals[s] = &x
				return vals[s]
			}
		}, adder, field.Name)
	}
}

type TreeEntry struct {
	csvEntry *Entry
	values   []idtrees.Value
	class    idtrees.Value
}

func (t *TreeEntry) Class() idtrees.Value {
	return t.class
}

func (t *TreeEntry) FieldValues() []idtrees.Value {
	return t.values
}

// A StringPointerValue is a Value that encapsulates a string,
// but unlike idtrees.StringValue, it does not force idtrees
// to use string comparisons each time it compares Value interfaces.
// Using this is beneficial for performance.
type StringPointerValue string

func (s *StringPointerValue) String() string {
	return string(*s)
}
