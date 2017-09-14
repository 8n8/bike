-- Copyright 5-o (c) 2017
-- All rights reserved

import qualified Test.Tasty as TT
import qualified TestObstacleMap as Om

main :: IO ()
main = TT.defaultMain tests

tests :: TT.TestTree
tests = TT.testGroup "" Om.tests 
