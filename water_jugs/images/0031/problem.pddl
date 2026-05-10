
(define (problem water-jug-pouring)
  (:domain waterjug)

  (:objects
     j1 j2 j3 - jug
  )

  (:init
    (= (total-pour) 0)

    ;; Capacity of each jugs 
    (= (capacity j1) 10) 
    (= (capacity j2) 3) 
    (= (capacity j3) 1) 

    ;; Intial water filled in each jugs 
    (= (contains j1) 8) 
    (= (contains j2) 2) 
    (= (contains j3) 1) 
) 


  (:goal
    (and 
      (= (contains j1) 9) 
      (= (contains j2) 1) 
      (= (contains j3) 1) 

    )
  )
  (:metric minimize (total-pour))
)
