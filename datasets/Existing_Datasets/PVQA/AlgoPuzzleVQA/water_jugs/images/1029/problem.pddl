
(define (problem water-jug-pouring)
  (:domain waterjug)

  (:objects
     j1 j2 j3 j4 j5 - jug
  )

  (:init
    (= (total-pour) 0)

    ;; Capacity of each jugs 
    (= (capacity j1) 11) 
    (= (capacity j2) 10) 
    (= (capacity j3) 9) 
    (= (capacity j4) 5) 
    (= (capacity j5) 4) 

    ;; Intial water filled in each jugs 
    (= (contains j1) 0) 
    (= (contains j2) 7) 
    (= (contains j3) 3) 
    (= (contains j4) 5) 
    (= (contains j5) 2) 
) 


  (:goal
    (and 
      (= (contains j1) 7) 
      (= (contains j2) 3) 
      (= (contains j3) 3) 
      (= (contains j4) 0) 
      (= (contains j5) 4) 

    )
  )
  (:metric minimize (total-pour))
)
