module TestObstacleMap (tests) where

-- import qualified ObstacleMap as Om
import qualified Test.Tasty as TT
import qualified Test.Tasty.HUnit as HU

tests :: [TT.TestTree]
tests =
    [ makeMap ]

makeMap :: TT.TestTree
makeMap = TT.testGroup "makeMap"
    [ HU.testCase "" $
        error "makeMap test not implemented yet" ]
