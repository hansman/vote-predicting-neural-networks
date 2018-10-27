#!/bin/env node

const fs = require('fs');
const { mepIds, orgIds } = require('./ids.json');
const { positions } = require('./positions.json');
const { votes } = require('./votes.json');
const _ = require('lodash');


mepIds.forEach((mep) => {

  const { mepId } = mep;
  console.log(`Parsing data set for mep ${mepId}`);

  const labels = [];
  const samples = [];

  const mepVotes = _.filter(votes, (v) => { return v.mep_id == mepId; });

  // orgs that voted on bills that `mepId` voted on
  const activeOrgs = [];
  mepVotes.forEach((v) => {
    const billId = v.bill_id;
    orgIds.forEach((orgId) => {
      const index = _.findIndex(positions, {organization_id: orgId, bill_id: billId});
      if (index != -1) {
        activeOrgs.push(orgId);
      }
    });
  });

  mepVotes.forEach((v) => {
    if (v.vote == 'for') {
      labels.push([0,0,1]);
    } else if (v.vote == 'abstain') {
      labels.push([0,1,0]);
    } else if (v.vote == 'against') {
      labels.push([1,0,0]);
    }
    const billId = v.bill_id;
    const features = [];

    // get orgs vote, aka position, for each active org
    activeOrgs.forEach((orgId) => {
      const index = _.findIndex(positions, {organization_id: orgId, bill_id: billId});
      if (index == -1) {
        features.push(0);
      } else {
        const p = positions[index];
        features.push(p.pro == true ? 1 : -1);
      }
    });

    samples.push(features);
    fs.writeFileSync(`./samples/${mep.mepId}.json`, JSON.stringify({labels, samples}));
  });

});
