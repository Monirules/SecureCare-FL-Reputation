// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;


contract Reputation {
    // Mapping of user address to reputation score (scaled as 1e18 for precision)
    mapping(address => uint256) public reputations;


    event ReputationUpdated(address indexed user, uint256 oldRep, uint256 newRep);


    function updateReputation(address user, uint256 score) external {
        require(score <= 1e18, "Score must be <= 1.0 (1e18)");

        uint256 old = reputations[user];

        uint256 newRep;
        if (old == 0) {
            newRep = score;
        } else {
            newRep = (old + score) / 2;
        }

        reputations[user] = newRep;
        emit ReputationUpdated(user, old, newRep);
    }

 
    function getReputation(address user) external view returns (uint256) {
        return reputations[user];
    }
}
