(define (problem wood-slide-puzzle)
  (:domain woodslide)

  (:objects
    b1 b2 b3 b4 b5 b6 b7 b8 b9 - block
    e1 e2 - empty-space
  )

  (:init
    (= (total-moves) 0)
    (= (min_x) 1) (= (min_y) 1)
    (= (max_x) 5) (= (max_y) 4)

    (= (xmin b1) 3) (= (ymin b1) 3)
    (= (xmax b1) 4) (= (ymax b1) 4)

    (= (xmin b2) 1) (= (ymin b2) 3)
    (= (xmax b2) 1) (= (ymax b2) 4)

    (= (xmin b3) 2) (= (ymin b3) 3)
    (= (xmax b3) 2) (= (ymax b3) 4)

    (= (xmin b4) 1) (= (ymin b4) 1)
    (= (xmax b4) 1) (= (ymax b4) 1)

    (= (xmin b5) 2) (= (ymin b5) 1)
    (= (xmax b5) 2) (= (ymax b5) 1)

    (= (xmin b6) 3) (= (ymin b6) 1)
    (= (xmax b6) 4) (= (ymax b6) 1)

    (= (xmin b7) 1) (= (ymin b7) 2)
    (= (xmax b7) 2) (= (ymax b7) 2)

    (= (xmin b8) 5) (= (ymin b8) 1)
    (= (xmax b8) 5) (= (ymax b8) 2)

    (= (xmin b9) 5) (= (ymin b9) 3)
    (= (xmax b9) 5) (= (ymax b9) 4)

    (= (x e1) 3) (= (y e1) 2)
    (= (x e2) 4) (= (y e2) 2)
    (= (dim_type b1) 4)
    (= (dim_type b2) 2)
    (= (dim_type b3) 2)
    (= (dim_type b4) 1)
    (= (dim_type b5) 1)
    (= (dim_type b6) 3)
    (= (dim_type b7) 3)
    (= (dim_type b8) 2)
    (= (dim_type b9) 2)

  )

  (:goal
    (and
      (= (xmin b1) 3) (= (ymin b1) 3)
      (= (xmax b1) 4) (= (ymax b1) 4)

      (= (xmin b2) 1) (= (ymin b2) 3)
      (= (xmax b2) 1) (= (ymax b2) 4)

      (= (xmin b3) 2) (= (ymin b3) 3)
      (= (xmax b3) 2) (= (ymax b3) 4)

      (= (xmin b4) 1) (= (ymin b4) 2)
      (= (xmax b4) 1) (= (ymax b4) 2)

      (= (xmin b5) 1) (= (ymin b5) 1)
      (= (xmax b5) 1) (= (ymax b5) 1)

      (= (xmin b6) 3) (= (ymin b6) 1)
      (= (xmax b6) 4) (= (ymax b6) 1)

      (= (xmin b7) 3) (= (ymin b7) 2)
      (= (xmax b7) 4) (= (ymax b7) 2)

      (= (xmin b8) 5) (= (ymin b8) 1)
      (= (xmax b8) 5) (= (ymax b8) 2)

      (= (xmin b9) 5) (= (ymin b9) 3)
      (= (xmax b9) 5) (= (ymax b9) 4)

    )
  )

  (:metric minimize (total-moves))
)
