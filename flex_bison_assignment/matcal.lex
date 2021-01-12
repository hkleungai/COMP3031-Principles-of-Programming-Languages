%option noyywrap
%{
#define YYSTYPE void*
#include "matcal.tab.h"
%}

/* Flex definitions */
whitespace                [ \t]+
newline                   [\n]
integer                   [0-9]+
add                       [+]
minus                     [-]
multiply                  [*]
left_round_bracket        [(]
right_round_bracket       [)]
left_square_bracket       [[]
right_square_bracket      []]
semi_colon                [;]
comma                     [,]


%%
{add}                               { return ADD; }
{minus}                             { return MINUS; }
{multiply}                          { return MULTIPLY; }
{left_round_bracket}                { return LEFT_ROUND_BRACKET;  }
{right_round_bracket}               { return RIGHT_ROUND_BRACKET; }
{left_square_bracket}               { return LEFT_SQUARE_BRACKET; }
{right_square_bracket}              { return RIGHT_SQUARE_BRACKET;  }
{semi_colon}                        { return SEMI_COLON;  }
{comma}                             { return COMMA; }
{integer}                           { yylval = (void*)atol(yytext); return T_INT; }
{newline}                           { return T_NL; }
{whitespace}                        /* ignore white spaces */
%%
