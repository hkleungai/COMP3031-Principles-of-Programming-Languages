(* Question 1 *)

val sumDigits = (
  fn array => List.foldr op+ 0 array
) o (
  fn integer_input =>
    List.map
      (fn (x: char) => Char.ord(x) - Char.ord(#"0"))
      (String.explode(Int.toString(integer_input)))
);

(* Question 2 *)

val frequencyPrefixSum = fn (array: int list, element: int) =>
  List.map
    (
      fn subarray =>
        List.length(
          List.filter (fn item => item = element) (subarray)
        )
    )
    ((
      fn (array) =>
        List.map
          (fn index => List.take(array, index))
          (List.tabulate(List.length(array), fn x => x + 1))
    )(array))
;

(* Question 3-5 helpers *)

datatype 'a llist = LList of 'a llist list| Elem of 'a;

val unleash__llist = fn (item) =>
  (
      fn  (LList element) => element
  )(
    (
      fn  (Elem element)  => LList([Elem(element)])
      |   (LList element) => LList element
    )((item))
  )
;

val flatten__n__times = fn n =>
  List.foldr
    (op o)
    (fn e => e)
    (
      List.tabulate(
        n,
        fn _ =>
          fn item =>
          LList(
            (
              fn array =>
                List.foldr
                (
                  fn  (Elem head__element, tail__element)   => [Elem(head__element)] @ tail__element
                  |   (LList head__element, tail__element)  => unleash__llist(LList(head__element)) @ tail__element
                )
                []
                array
            )(unleash__llist(item))
          )
      )
    )
;

(* Question 3 *)

val flatten = fn object =>
  let
    fun flatten__as__many__as__needed(item, []) = item
      | flatten__as__many__as__needed(item, head_element::tail_subarray) =
        let
          val n__flattened__llist = flatten__n__times(head_element)(item)
          val unleashed__n__flattened__llist = unleash__llist(n__flattened__llist)
        in
          if List.length(
            List.filter (fn (Elem element) => true | _ => false) unleashed__n__flattened__llist
          ) = List.length(unleashed__n__flattened__llist)
            then n__flattened__llist
            else flatten__as__many__as__needed(item, tail_subarray)
        end
  in
    (
      fn array => List.map (fn (Elem element) => element) array
    )(
      unleash__llist(
        flatten__as__many__as__needed(object, List.tabulate(1000, fn n => n))
      )
    )
  end
;

(* Question 4 *)

fun depth(Elem object) = 0
  | depth(LList object) =
      let
        fun flatten__as__many__as__needed(item, []) = 1
          | flatten__as__many__as__needed(item, head_element::tail_subarray) =
            let
              val n__flattened__llist = flatten__n__times(head_element)(LList item)
              val unleashed__n__flattened__llist = unleash__llist(n__flattened__llist)
            in
              if List.length(
                List.filter (fn (Elem element) => true | _ => false) unleashed__n__flattened__llist
              ) = List.length(unleashed__n__flattened__llist)
                then head_element + 1
                else flatten__as__many__as__needed(item, tail_subarray)
            end
      in
        flatten__as__many__as__needed(object, List.tabulate(1000, fn n => n))
      end
;

(* Question 5 *)

fun equal(Elem a, Elem b) = (op =)(Elem a, Elem b)
  | equal(LList a, LList b) = (op =)(LList a, LList b)
  | equal(Elem a, LList b) = false
  | equal(LList a, Elem b) = false
;
