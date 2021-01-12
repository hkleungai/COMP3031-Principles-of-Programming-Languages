# do NOT modify this file
CC = gcc
CXX = g++
LEX = flex
YACC = bison
RM = rm

.PHONY: clean

matcal: matcal.c matcal.tab.c helpers.c
	$(CC) -o $@ $^

matcal.c: matcal.lex
	$(LEX) -o $@ $<

matcal.tab.c: matcal.y
	$(YACC) -d $<

clean:
	$(RM) -rf matcal.c matcal matcal.tab.*
