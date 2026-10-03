from fractions import Fraction
from itertools import combinations

import pytest

from vlm.rules.calculator import (matrix_cost, settle_matrix, power_jackpot_distribution,
                                 max3d_cost, settle_max3d, settle_keno, settle_bingo18, bingo18_catalog)


@pytest.mark.parametrize('game,n,plays,cost',[
    ('mega645',5,40,400000),('power655',5,50,500000),('mega645',7,7,70000),
    ('power655',7,7,70000),('mega645',18,18564,185640000),('power655',18,18564,185640000)])
def test_bao_price(game,n,plays,cost):
    result=matrix_cost(game,list(range(1,n+1)))
    assert (result.plays,result.cost)==(plays,cost)


@pytest.mark.parametrize('n',[4,16,17,19])
def test_unsupported_system_size_is_rejected(n):
    with pytest.raises(ValueError): matrix_cost('mega645',list(range(1,n+1)))


def test_bao7_prize_matrix():
    r=settle_matrix('mega645',list(range(1,8)),[1,2,3,4,5,6])
    assert r.prize_counts=={'jackpot1':1,'first':6}
    assert r.fixed_payout==60000000
    assert r.jackpot_payout is None


def test_mega_bao5_awards_every_complement_play():
    r=settle_matrix('mega645',[1,2,3,4,5],[1,2,3,4,5,6])
    assert r.prize_counts=={'jackpot1':1,'first':39}
    assert r.fixed_payout==390000000


@pytest.mark.parametrize('game,n,h,include_bonus',[
    ('mega645',5,3,False),('mega645',7,4,False),('mega645',18,5,False),
    ('power655',5,3,True),('power655',7,4,True),('power655',18,5,True),
    ('power655',18,6,True),('power655',18,6,False),
])
def test_bao_prize_counts_match_independent_ticket_expansion(game,n,h,include_bonus):
    from collections import Counter
    chosen=list(range(1,h+1))+([7] if include_bonus else [])
    chosen+=list(range(8,8+n-len(chosen)))
    plays=(combinations(chosen,6) if n>=6 else
           (tuple(chosen+[x]) for x in range(1,(55 if game=='power655' else 45)+1) if x not in chosen))
    expected=Counter()
    for play in plays:
        matches=len(set(play)&{1,2,3,4,5,6})
        tier=('jackpot1' if matches==6 else 'jackpot2' if game=='power655' and matches==5 and 7 in play
              else 'first' if matches==5 else 'second' if matches==4 else 'third' if matches==3 else None)
        if tier: expected[tier]+=1
    kwargs={'bonus_number':7} if game=='power655' else {}
    assert settle_matrix(game,chosen,[1,2,3,4,5,6],**kwargs).prize_counts==dict(expected)


def test_bao18_prize_matrix_and_bruteforce():
    r=settle_matrix('mega645',list(range(1,19)),[1,2,3,4,5,6])
    assert r.prize_counts=={'jackpot1':1,'first':72,'second':990,'third':4400}
    brute={6:0,5:0,4:0,3:0}
    for t in combinations(range(1,19),6):
        hits=len(set(t)&{1,2,3,4,5,6})
        if hits>=3: brute[hits]+=1
    assert [brute[x] for x in [6,5,4,3]]==[1,72,990,4400]


def test_bao5_power_with_and_without_bonus():
    r=settle_matrix('power655',[1,2,3,4,5],[1,2,3,4,5,6],bonus_number=7)
    assert r.prize_counts=={'jackpot1':1,'jackpot2':1,'first':48}
    r=settle_matrix('power655',[1,2,3,4,7],[1,2,3,4,5,6],bonus_number=7)
    assert r.prize_counts=={'jackpot2':2,'second':48}


def test_power_shared_pots_and_transfer():
    p=power_jackpot_distribution(367000000000,40000000000,0,2)
    assert p.transfer==67000000000
    assert p.per_winner2==53500000000
    assert p.next_base1==300000000000
    assert power_jackpot_distribution(367000000000,40000000000,2,2).transfer==0
    r=settle_matrix('power655',[1,2,3,4,5,6,7],[1,2,3,4,5,6],bonus_number=7,
                    jackpot_pots={'jackpot1':30000000000,'jackpot2':3000000000},
                    total_jackpot_winners={'jackpot1':2,'jackpot2':10})
    assert r.prize_counts=={'jackpot1':1,'jackpot2':6}
    assert r.jackpot_payout==Fraction(16800000000)
    assert r.fixed_payout==0


def test_power_published_payable_pool_is_not_transferred_twice():
    ticket=[1,2,3,4,5,7]
    draw=[1,2,3,4,5,6]
    r=settle_matrix('power655',ticket,draw,bonus_number=7,
                    jackpot_pots={'jackpot1':367000000000,'jackpot2':107000000000},
                    total_jackpot_winners={'jackpot1':0,'jackpot2':2})
    assert r.jackpot_payout==53500000000
    pre=settle_matrix('power655',ticket,draw,bonus_number=7,jackpot_pots_basis='pre_transfer',
                      jackpot_pots={'jackpot1':367000000000,'jackpot2':40000000000},
                      total_jackpot_winners={'jackpot1':0,'jackpot2':2})
    assert pre.jackpot_payout==r.jackpot_payout


def test_research_wildcard_expansion_can_be_explicitly_requested():
    assert max3d_cost('max3d','bao_vi_tri',['1*3'],allow_unverified=True).plays==10


def test_max_permutations_and_cumulative_prizes():
    assert max3d_cost('max3dpro','bao_bo_so',['112','000']).plays==3
    assert max3d_cost('max3dpro','bao_nhieu_bo_so',['111','222','333']).plays==6
    draw=['123','456','123','456','111','222']+['789']*6+['999']*8
    r=settle_max3d('max3dpro','co_ban',['123','456'],draw)
    assert r.prize_counts=={'dac_biet':1,'nhat':1,'tu':1,'nam':2,'sau':2}
    assert r.fixed_payout==2031280000


def test_max_pro_official_two_ordered_plays_have_four_fifth_prizes():
    # Vietlott's 31/01/2026 announcement: 15 each special/aux, 30 fourth, 60 fifth.
    draw=['123','456']+['789']*18
    one=settle_max3d('max3dpro','co_ban',['123','456'],draw,stake_multiple=15)
    two=settle_max3d('max3dpro','co_ban',['456','123'],draw,stake_multiple=15)
    assert one.prize_counts['nam']+two.prize_counts['nam']==60
    assert one.fixed_payout+two.fixed_payout==36036000000


def test_existing_max_engine_counts_each_matching_number(monkeypatch):
    import numpy as np
    from vietlott_engine.game_theory import max3d
    draw=np.asarray([[123,456]+[789]*18])
    monkeypatch.setattr(max3d,'simulate_draws',lambda n,rng:np.repeat(draw,n,axis=0))
    assert max3d.simulate_plays(max3d.MAX3D_PRO,[(123,456)],sims=1)[0]==2001200000
    dist=max3d.play_distribution(max3d.MAX3D_PRO,(123,456))
    assert any(pay==2001200000 and probability>0 for pay,probability in dist.distribution)


def test_unverified_max_bao_is_not_silently_sold():
    with pytest.raises(ValueError): max3d_cost('max3d','bao_bo_so',['123'])
    assert max3d_cost('max3d','bao_bo_so',['123'],allow_unverified=True).plays==6


def test_keno_all_bacs_zero_hits_and_cap():
    for bac in range(1,11):
        r=settle_keno(list(range(1,bac+1)),list(range(1,21)))
        assert r.fixed_payout>0
    assert settle_keno(list(range(21,30)),list(range(1,21))).fixed_payout==10000
    r=settle_keno(list(range(1,11)),list(range(1,21)),total_top_winning_units=6)
    assert r.fixed_payout==Fraction(10000000000,6)
    r=settle_keno(list(range(1,9)),list(range(1,21)),total_top_winning_units=100)
    assert r.fixed_payout==100000000


@pytest.mark.parametrize('bet,draw,payout',[
    ('big',list(range(41,54))+list(range(1,8)),26000),
    ('big_small_tie',list(range(1,11))+list(range(41,51)),26000),
    ('even_11_12',list(range(2,25,2))+list(range(1,16,2)),20000),
    ('even',list(range(2,25,2))+list(range(1,16,2)),0)])
def test_keno_side_boundaries(bet,draw,payout):
    assert settle_keno([],draw,side_bet=bet).fixed_payout==payout


def test_bingo_all_38_bets_and_repeat():
    assert len(bingo18_catalog())==38
    assert settle_bingo18('single',1,[1,1,1]).fixed_payout==30000
    assert settle_bingo18('double',1,[1,1,1]).fixed_payout==75000
    assert settle_bingo18('triple',1,[1,1,1]).fixed_payout==1200000
    assert settle_bingo18('any_triple',None,[1,1,1]).fixed_payout==200000
    assert settle_bingo18('sum',18,[6,6,6]).fixed_payout==1200000
    assert settle_bingo18('big',None,[6,6,6]).fixed_payout==15000


@pytest.mark.parametrize('stake',[0,-1,1.5,True])
def test_invalid_stake_rejected(stake):
    with pytest.raises(ValueError): settle_bingo18('sum',3,[1,1,1],stake_multiple=stake)
