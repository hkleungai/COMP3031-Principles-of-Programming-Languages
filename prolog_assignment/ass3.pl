/* PART I BEGINS */

/* Q1 BEGINS */

% Compute linear regression by the least-squares approach.
linear_regression(A, B) :-
  % Extract X, Y from xs, ys.
  xs(X), ys(Y),
  % Apply length() to obtain N, check if dimensions match.
  length(X, N), length(Y, N),
  % Apply sum_list() to obtain SX, SY.
  sum_list(X, SX), sum_list(Y, SY),
  % Apply maplist() with square-lambda to X to obtain LXX.
  % Apply sum_list() to obtain SXX.
  maplist([A, A * A] >> true, X, LXX), sum_list(LXX, SXX),
  % Apply maplist() with cross-product-lambda to X, Y to obtain LXY.
  % Apply sum_list() to obtain SXY.
  maplist([A, B, A * B] >> true, X, Y, LXY), sum_list(LXY, SXY),
  % Compute slope by the given formula.
  A is (N * SXY - SX * SY) / (N * SXX - SX * SX),
  % Compute y-intercept by the given formula.
  B is (SY * SXX - SX * SXY) / (N * SXX - SX * SX).

/* Q1 ENDS */

/* Q2 BEGINS */

% Determine adjacency for each pair of vertices
adjacent_edges(X, Y) :- edge(X, Y) ; edge(Y, X).

% Custom helper for reducing list of lists.
% @param {lambda} F
%   A lambda for comparing current and accumulated items to determine whether the current item would appear in the new list
% @param {list} P
%   Input
% @param {list} R
%   Output
process_list(F, P, R) :-
  foldl([CR, AC, TA] >> (
    call(F, CR, AC) -> (TA = AC) ; append(AC, [CR], TA)
  ), P, [], R).

% Determine whether the input vertex is dense.
dense(R) :-
  % Step 1: Obtain the list of dense vertices, most likely with duplication.
  findall(V, (
    adjacent_edges(V, A), adjacent_edges(V, B), adjacent_edges(V, C),
    dif(A, B), dif(B, C), dif(C, A)
  ), LLDR),
  % Step 2: Reduce the list by builtin member(),
  % so that if the current item has already appeared in accumulated list,
  % it would not be further included during the reduction.
  process_list(member, LLDR, LDR),
  % Step 3: Change the list result back to a relation-check builtin member().
  member(R, LDR).

% Obtain all vertices by the given adjacency, and remove possible duplications
vertices(R) :-
  findall(V, adjacent_edges(V, _), RR),
  process_list(member, RR, R).

% Compute, by the length of edge-list N, a list in form of [1..N]
num_edge_list(R) :- findall(V, edge(V, _), L), length(L, LEN), numlist(1, LEN, R).

% Initialize the path computation of two vertices.
% path_cycle() is called with 6 args, in which the 5th is an empty visited list.
% @param {enum} FLAG
%   path | cycle
% @param {lambda} ADJACENT
%   a lambda relation for determining vertex adjacency
% @param {vertex} A
%   vertex_1
% @param {vertex} B
%   vertex_2
% @param {vertex-list} VT
%   (NOT in the 1st clause) a list storing visited vertex in the current path.
% @param {vertex} R
%   resultant path
path_cycle(FLAG, ADJACENT, A, B, R) :- path_cycle(FLAG, ADJACENT, A, B, [], R).
% Finishing the path computation of two vertices.
% Here add the end-point to the path result.
path_cycle(path, _, B, B, _, R) :- R = [B].
% Finishing the cycle computation of two vertices
% Here skip the end-point for the cycle result.
path_cycle(cycle, _, B, B, _, R) :- R = [].
% Compute the subpath / sub-cycle A - NX - ... - B,
% If A is not visited nor the end-point, obtain its neighbor NX and then
% - add A to visited list VT
% - recurse on NX - ... - B
% Notice for path, A and B should be distinct.
% But for cycle, A and B can be the same at the first step.
path_cycle(FLAG, ADJACENT, A, B, VT, R) :-
  % Suppose A is not yet visited
  \+ member(A, VT),
  % Make sure A and B are distinct at certain steps to make sure the clause terminates properly.
  % For FLAG == path, we should not have them equal in EACH AND EVERY part of the computation.
  % For FLAG == cycle, we can only allow them to be equal at the first step, i.e. when VT is []
  (FLAG == path -> dif(A, B) ; (length(VT, 0) ; dif(A, B))),
  % From A visit its neighbor NX
  % Add A to the list of visited vertices, store the whole thing to NVT
  % Recursively put NX - ... - B with NVT into the clause,
  call(ADJACENT, A, NX), NVT = [A|VT], path_cycle(FLAG, ADJACENT, NX, B, NVT, TA),
  % Finally gather the result and return
  R = [A|TA].

% Compute the shortest path btw two distinct vertices.
% @param {vertex} A
%   vertex_1
% @param {vertex} B
%   vertex_2
% @param {vertex} R
%   resultant path
shortest_path(A, B, R) :-
  % Step 1: Obtain each valid path from A to B and store it to LSP.
  findall(X, path_cycle(path, adjacent_edges, A, B, X), LSP),
  % Step 2: Sort each valid paths by their length.
  % - Apply map_list_to_pairs() with length() on LSP to obtain a sortable NLSP.
  % - Apply keysort() on NLSP to obtain SNLSP.
  % - Apply pairs_values() on SNLSP to retrieve values and store it to SLSP.
  map_list_to_pairs(length, LSP, NLSP), keysort(NLSP, SNLSP), pairs_values(SNLSP, SLSP),
  % Step 3: Pick the 1st item in the sorted SLSP and store it to R
  nth1(1, SLSP, R).

% Determine all the shortest path from a sparse vertex
% to a dense vertex with certain path-length
% @param {vertex} A
%   a sparse vertex
% @param {int} PLEN
%   a valid path-length
% @param {vertex-list-list} R
%   a list of path
pointwise_sparse_dense_path(A, PLEN, R) :- (
  findall(X, (
    \+ dense(A), dense(B),
    shortest_path(A, B, X),
    VNUM is PLEN + 1, length(X, VNUM)
  ), R)
).

% Determine the list of interesting vertices.
interesting_list(R) :-
  % Step 1: Obtain from number of edges N a list [1..N] and the list of vertices V.
  num_edge_list(N), vertices(V),
  % % Step 2: Iterare for possible path-lengths to get all interesting paths and store them in DLLLINT.
  findall(LPDP, (
    % Step 2.1: Compute the list of sparse-dense-path, LLPDP,
    % on each vertex given the current path-length NN.
    member(NN, N), maplist([VV, RRR] >> pointwise_sparse_dense_path(VV, NN, RRR), V, LLPDP),
    % Step 2.2: Reduce LPDP to LPDP by filtering on the number of sparse-dense-path on each vertex.
    include(([PDP] >> (length(PDP, LENPDP), LENPDP >= 2)), LLPDP, LPDP), \+ length(LPDP, 0)
  ), LLLINT),
  % Step 3: Pick from LLLINT the paths with shortest path-lengths and store it to LLINT.
  nth1(1, LLLINT, LLINT),
  % Step 4: Pick from each vertex in LLINT the first path and perform maplist to obtain LINT.
  maplist(([ILINT, RLINT] >> nth1(1, ILINT, RLINT)), LLINT, LINT),
  % Step 5: Pick from each vertex-path in LINT the first vertex and perform maplist to obtain R.
  maplist(([IINT, RINT] >> nth1(1, IINT, RINT)), LINT, R).

% Determine the interesting relation by applying member() to L.
interesting(R) :- interesting_list(L), member(R, L).

/* Q2 ENDS */

/* PART I ENDS */

/* PART II BEGINS */

% Retrieve bond-list from atom.
element_bond(C, R) :- atom_elements(C, _, R).
% Match atom's element-type with self.
type_atom(TP, AT) :- atom_elements(AT, TP, _).

/* Q3 BEGINS */

% Obtain list of carbons whose bond list is composed of 1 C and 3 H.
ch3(R) :- (
  % Apply findall() to locate the required atoms
  findall(C, (
    % Step 1: Check if the atom is carbon and retrieve its bond-list L.
    type_atom(carbon, C), element_bond(C, L),
    % Step 2: Check for the length of L.
    length(L, 4),
    % Step 3: Filter carbons and hydrogens in two list and check for the list-lengths.
    include(type_atom(carbon), L, LC), length(LC, 1),
    include(type_atom(hydrogen), L, LH), length(LH, 3)
  ), R)
).

/* Q3 ENDS */

/* Q4 BEGINS */

% A relation for adjacent-carbons.
% Here adjacency means one is in another bond-list
% Lazy implementation:
%   Assume the given facts are all well-defined
%   so only one-sided check is performed.
adjacent_carbons(X, Y) :-
  type_atom(carbon, X), type_atom(carbon, Y),
  element_bond(X, XB), member(Y, XB).

% Determine possible c6rings.
% In this relation multiple list element could be describing the same cycle.
% Each valid cycle brings 6 * 2 = 12 rotations.
c6ring_rotation(R) :-
  type_atom(carbon, C),
  path_cycle(cycle, adjacent_carbons, C, C, R),
  length(R, 6).

% Determine if the current cycle is contained in the accumulated list.
% @param {vertex-list} X
%   a list of vertex representing a cycle.
% @param {vertex-list-list} L
%   a list of vertex-list representing the accumulated cycles.
% Here determine whether the two are permutations of each other using msort.
is_in_carbons_cycle_list(X, L) :-
  include([CYCLE] >> (msort(X, Y), msort(CYCLE, Y)), L, LL), \+ length(LL, 0).

% Determine the list of c6ring
c6ring(R) :-
  % Step 1: Obtain the list of c6ring rotations
  findall(X, c6ring_rotation(X), CSIX),
  % Step 2: Reduce the rotations with is_in_carbons_cycle_list()
  process_list(is_in_carbons_cycle_list, CSIX, R).

/* Q4 ENDS */

/* Q5 BEGINS */

% A relation to link up a carbon and its NO2 bond, if any.
carbon_NO2_bondings(C, R) :-
  % Step 1:
  % - Obtain carbon C and its bond list B.
  % - Check if B contains exactly one nitrogen, and store it to N if yes.
  type_atom(carbon, C), element_bond(C, B),
  include(type_atom(nitrogen), B, LN), length(LN, 1), nth1(1, LN, N),
  % Step 2:
  % - Obtain the bond-list CO for N.
  % - Check if B contains exactly two oxygen, and filter it to LO if yes.
  element_bond(N, CO), include(type_atom(oxygen), CO, LO), length(LO, 2),
  % Step 3: Combine N and CO and store it to R.
  R = [N|LO].

% Deterine the list of tnt structure on a given bond graph.
tnt(R) :-
  % Step 1: Obtain all the c6ring rotations with NO2 at 2, 4, 6 and store it to DLTNT.
  findall(S, (
    c6ring_rotation(SIX),
    include([ISIX] >> (
      nth1(I, SIX, ISIX), IEO is mod(I, 2),
      IEO == 1 -> (\+ carbon_NO2_bondings(ISIX, _)) ; (carbon_NO2_bondings(ISIX, B), length(B, 3))
    ), SIX, S),
    length(S, 6)
  ), DLTNT),
  % Step 2: Reduce DLTNT by is_in_carbons_cycle_list(), and store the result to LTNT.
  process_list(is_in_carbons_cycle_list, DLTNT, LTNT),
  % Step 3: Map LTNT so that at each cycle, entries 2,4,6 are mapped to NO2 bond.
  findall(SS, (
    member(TNTSIX, LTNT),
    maplist([SSS, RR] >> (
      nth1(J, TNTSIX, SSS), JEO is mod(J, 2),
      JEO == 1 -> (RR = SSS) ; (carbon_NO2_bondings(SSS, B), RR = [SSS|B])
    ), TNTSIX, SS)
  ), R).

/* Q5 ENDS */

/* PART II ENDS */
