adj([a, [b,c,j]]).
adj([j, [g,b]]).
adj([c, [d]]).
adj([d, [e]]).
adj([e, [f]]).
adj([f, [h,g]]).

edge(V1, V2) :- directed_edge(V1, V2).
edge(V1, V2) :- directed_edge(V2, V1).

directed_edge(V1, V2) :- adj([H, T]), V1 == H, member(H2, T), V2==H2, !.

directed_edge(V1, V2) :- !, adj([V1, T]), member(V2, T).

directed_edge(V1, _) :- adj([V1, _]), fail, !.
