lex predToPy.l
bison -d predToPy.y
gcc structs.c predToPy.tab.c lex.yy.c -o Booleanize
