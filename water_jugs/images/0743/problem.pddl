
(define (problem water-jug-pouring)
  (:domain waterjug)

  (:objects
     j1 j2 j3 j4 - jug
  )

  (:init
    (= (total-pour) 0)

    ;; Capacity of each jugs 
    (= (capacity j1) 12) 
    (= (capacity j2) 7) 
    (= (capacity j3) 5) 
    (= (capacity j4) 3) 

    ;; Intial water filled in each jugs 
    (= (contains j1) 4) 
    (= (contains j2) 7) 
    (= (contains j3) 3) 
    (= (contains j4) 3) 
) 


  (:goal
    (and 
      (= (contains j1) 10) 
      (= (contains j2) 4) 
      (= (contains j3) 3) 
      (= (contains j4) 0) 

    )
  )
  (:metric minimize (total-pour))
)
