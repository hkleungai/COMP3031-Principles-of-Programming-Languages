%{
#define YYSTYPE void*
#include <stdio.h>
#include "helpers.h"
%}

/* Define tokens here */
%token
T_NL
T_INT
ADD
MINUS
MULTIPLY
LEFT_ROUND_BRACKET
RIGHT_ROUND_BRACKET
LEFT_SQUARE_BRACKET
RIGHT_SQUARE_BRACKET
SEMI_COLON
COMMA

%%
input:    /* empty */
    |     input line;

line:     T_NL
    |     expr T_NL { print_matrix($1); };

element:  T_INT { $$ = element2matrix((long)$1); };

expr:     expr ADD sub_expr { $$ = matrix_add($1, $3); }
    |     expr MINUS sub_expr { $$ = matrix_sub($1, $3); }
    |     sub_expr { $$ = $1; };

sub_expr: sub_expr MULTIPLY unit { $$ = matrix_mul($1, $3); }
    |     unit { $$ = $1; };

unit:     LEFT_ROUND_BRACKET expr RIGHT_ROUND_BRACKET { $$ = $2; }
    |     matrix { $$ = $1; };

matrix:   LEFT_SQUARE_BRACKET rows RIGHT_SQUARE_BRACKET { $$ = $2;};

rows:     rows SEMI_COLON row { $$ = append_row($1, $3); }
    |     row { $$ = $1; };

row:      row COMMA element { $$ = append_element($1, $3); }
    |     element { $$ = $1; };
%%

int main() { return yyparse(); }
int yyerror(const char* s) { printf("%s\n", s); return 0; }
