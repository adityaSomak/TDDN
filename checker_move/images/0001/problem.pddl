(define (problem checker-problem)
  (:domain checker-move)

  (:objects
    c1 c2 c3 - checker
    p1 p2 p3 p4 - position
  )

  (:init
    (checker-at c1 p1)
    (checker-at c2 p2)
    (checker-at c3 p4)
    (adjacent p1 p2)
    (adjacent p2 p3)
    (adjacent p3 p4)
    (adjacent p2 p1)
    (adjacent p3 p2)
    (adjacent p4 p3)

    (= (has-color c1) 0)
    (= (has-color c2) 0)
    (= (has-color c3) 1)

    (empty p3)
    (less p1 p2)
    (less p2 p3)
    (less p3 p4)
  )

  (:goal
    (and
      (checker-at c3 p1)
      (checker-at c1 p3)
      (checker-at c2 p4)
    )
  )
)