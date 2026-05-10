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

    (= (xmin b1) 2) (= (ymin b1) 2)
    (= (xmax b1) 3) (= (ymax b1) 3)

    (= (xmin b2) 1) (= (ymin b2) 1)
    (= (xmax b2) 1) (= (ymax b2) 2)

    (= (xmin b3) 1) (= (ymin b3) 3)
    (= (xmax b3) 1) (= (ymax b3) 4)

    (= (xmin b4) 2) (= (ymin b4) 4)
    (= (xmax b4) 2) (= (ymax b4) 4)

    (= (xmin b5) 3) (= (ymin b5) 4)
    (= (xmax b5) 3) (= (ymax b5) 4)

    (= (xmin b6) 3) (= (ymin b6) 1)
    (= (xmax b6) 4) (= (ymax b6) 1)

    (= (xmin b7) 4) (= (ymin b7) 2)
    (= (xmax b7) 5) (= (ymax b7) 2)

    (= (xmin b8) 4) (= (ymin b8) 3)
    (= (xmax b8) 4) (= (ymax b8) 4)

    (= (xmin b9) 5) (= (ymin b9) 3)
    (= (xmax b9) 5) (= (ymax b9) 4)

    (= (x e1) 2) (= (y e1) 1)
    (= (x e2) 5) (= (y e2) 1)
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
      (= (xmin b1) 2) (= (ymin b1) 1)
      (= (xmax b1) 3) (= (ymax b1) 2)

      (= (xmin b2) 1) (= (ymin b2) 1)
      (= (xmax b2) 1) (= (ymax b2) 2)

      (= (xmin b3) 1) (= (ymin b3) 3)
      (= (xmax b3) 1) (= (ymax b3) 4)

      (= (xmin b4) 3) (= (ymin b4) 3)
      (= (xmax b4) 3) (= (ymax b4) 3)

      (= (xmin b5) 3) (= (ymin b5) 4)
      (= (xmax b5) 3) (= (ymax b5) 4)

      (= (xmin b6) 4) (= (ymin b6) 1)
      (= (xmax b6) 5) (= (ymax b6) 1)

      (= (xmin b7) 4) (= (ymin b7) 2)
      (= (xmax b7) 5) (= (ymax b7) 2)

      (= (xmin b8) 4) (= (ymin b8) 3)
      (= (xmax b8) 4) (= (ymax b8) 4)

      (= (xmin b9) 5) (= (ymin b9) 3)
      (= (xmax b9) 5) (= (ymax b9) 4)

    )
  )

  (:metric minimize (total-moves))
)
