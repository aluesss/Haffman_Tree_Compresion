"""
Assignment 2 starter code
CSC148, Winter 2020
Instructors: Bogdan Simion, Michael Liut, and Paul Vrbik

This code is provided solely for the personal and private use of
students taking the CSC148 course at the University of Toronto.
Copying for purposes other than this use is expressly prohibited.
All forms of distribution of this code, whether as given or with
any changes, are expressly prohibited.

All of the files in this directory and all subdirectories are:
Copyright (c) 2020 Bogdan Simion, Michael Liut, Paul Vrbik, Dan Zingaro
"""
from __future__ import annotations
import time
from typing import Dict, Tuple, Union, Any, List
from utils import *
from huffman import HuffmanTree


# ====================
# Helper Function
def _sort_dict(freq_dict: Dict[int, int]) -> Dict[int, int]:
    """
    Sort a dictionary by its value

    >>> b = {2: 6, 3: 4, 7: 5}
    >>> _sort_dict(b)
    {3: 4, 7: 5, 2: 6}
    >>> b == {2: 6, 3: 4, 7: 5}
    True
    >>> c = {2: 6, 3: 4}
    >>> c = _sort_dict(c)
    >>> c == {3: 4, 2: 6}
    True
    """
    a1 = []
    ans = {}
    for i in freq_dict:
        a1.append(freq_dict[i])
        a1.sort()
    for a in a1:
        for b in freq_dict:
            if freq_dict[b] == a:
                ans[b] = freq_dict[b]
    return ans


def _get_two_mins(d: Dict[int, int]) -> Tuple[int, int]:
    """
    Get two smallest value's key in a sorted dictionary as tuple.
    >>> a = {3: 4, 7: 5, 2: 6}
    >>> _get_two_mins(a)
    (3, 7)
    >>> b = {3: 4, 2: 6}
    >>> _get_two_mins(b)
    (3, 2)
    """
    a = list(d.keys())
    return a[0], a[1]


def _set_symbols(tree: HuffmanTree, x: str) -> List[int]:
    """
    Create a dictionary contains huffman tree's value and code
    >>> tree2 = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> _set_symbols(tree2, '')
    {2: '0', 3: '10', 7: '11'}
    >>> tree = HuffmanTree(None, HuffmanTree(5), HuffmanTree(4))
    >>> _set_symbols(tree, '')
    {5: '0', 4: '1'}
    >>> tree3 = HuffmanTree(None, tree, tree2)
    >>> _set_symbols(tree3, '')
    {5: '00', 4: '01', 2: '10', 3: '110', 7: '111'}
    """
    ans = {}
    if tree.symbol is not None:
        ans[tree.symbol] = x
    else:
        ans.update(_set_symbols(tree.left, x + '0'))
        ans.update(_set_symbols(tree.right, x + '1'))

    return ans


def _set_node(tree: HuffmanTree) -> list:
    """
    Put all subtree whose symbol is None in a list.
    >>> tree2 = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> tree = HuffmanTree(None, HuffmanTree(5), HuffmanTree(4))
    >>> a = _set_node(tree)
    >>> a == [HuffmanTree(None, HuffmanTree(5), HuffmanTree(4))]
    True
    >>> b = _set_node(tree2)
    >>> b == [HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)), \
    HuffmanTree(None, HuffmanTree(2), HuffmanTree(None, HuffmanTree(3),\
     HuffmanTree(7)))]
    True
    """
    if tree.symbol is not None:
        return []
    else:
        ans = [tree]
        left = _set_node(tree.left)
        right = _set_node(tree.right)
        return left + right + ans


def _generate_tree_postorder_helper(copy_list: List[ReadNode]) -> HuffmanTree:
    """
    Build a HuffmanTree by a list of ReadNode.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> _generate_tree_postorder_helper(lst)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    >>> lst2 = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(0, 6, 1, 0), ReadNode(1, 0, 1, 0)]
    >>> _generate_tree_postorder_helper(lst2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), HuffmanTree(None, HuffmanTree(6, None, None), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None))))
    """
    node = copy_list[-1]
    copy_list.pop()
    if node.l_type == 1 and node.r_type == 1:
        root = HuffmanTree(None)
        root.right = _generate_tree_postorder_helper(copy_list)
        root.left = _generate_tree_postorder_helper(copy_list)
    elif node.l_type == 0 and node.r_type == 0:
        root = HuffmanTree(None, HuffmanTree(node.l_data),
                           HuffmanTree(node.r_data))
    elif node.l_type == 0:
        root = HuffmanTree(None, HuffmanTree(node.l_data), None)
        root.right = _generate_tree_postorder_helper(copy_list)
    else:
        root = HuffmanTree(None, None, HuffmanTree(node.r_data))
        root.left = _generate_tree_postorder_helper(copy_list)

    return root


def _switch_dict(dic: Dict[Any, Any]) -> Tuple[dict, List[int]]:
    """
    Switch a dictionary's key and value
    >>> d = {1: 'a', 2: 'b', 3: 'c'}
    >>> _switch_dict(d)
    ({'a': 1, 'b': 2, 'c': 3}, [1])
    >>> d2 = {104: '000', 101: '001', 119: '010', 114: '011', 108: '10', \
100: '110', 111: '111'}
    >>> _switch_dict(d2)
    ({'000': 104, '001': 101, '010': 119, '011': 114, '10': 108, '110': 100, \
'111': 111}, [3, 2])
    """
    ans = {}
    lst = []
    for i in dic:
        ans[dic[i]] = i
        if len(dic[i]) not in lst:
            lst.append(len(dic[i]))
    return ans, lst


def _decompress_bytes_helper(bits: str, dic: Dict[str, int],
                             size: int, len_list: List[int]) \
        -> Tuple[List[int], str]:
    """
    Return a list of integers that is built by bits and dic, length of list
    equal to size

    >>> s = '00000110101110101110111011000000'
    >>> d = {'000': 104, '001': 101, '010': 119, '011': 114, '10': 108, \
    '110': 100, '111': 111}
    >>> len_list = [3, 2]
    >>> _decompress_bytes_helper(s, d, 10, len_list)
    ([104, 101, 108, 108, 111, 119, 111, 114, 108, 100], '00000')
    """
    ans = []
    b = bits
    if size > 0:
        for i in len_list:
            if bits[:i] in dic:
                b = bits[i:]
                ans.append(dic[bits[:i]])
                size = size - 1
                t = _decompress_bytes_helper(b, dic, size, len_list)
                ans.extend(t[0])
                b = t[1]
                break
    return ans, b


def _list_leaves(tree: HuffmanTree) -> List[HuffmanTree]:
    """
    Return a list contains all the leaves in tree.
    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> _list_leaves(tree)
    [HuffmanTree(99, None, None), HuffmanTree(100, None, None), \
HuffmanTree(101, None, None), HuffmanTree(97, None, None), \
HuffmanTree(98, None, None)]
    """
    if isinstance(tree.symbol, int):
        return [tree]
    else:
        ans = []
        left = _list_leaves(tree.left)
        right = _list_leaves(tree.right)
        return left + ans + right


def _sort_dict_reverse(freq_dict: Dict[int, int]) -> Dict[int, int]:
    """
    Sort a dictionary by its value from highest to lowest

    >>> b = {2: 6, 3: 4, 7: 5}
    >>> _sort_dict_reverse(b)
    {2: 6, 7: 5, 3: 4}
    >>> b == {2: 6, 3: 4, 7: 5}
    True
    """
    a1 = []
    ans = {}
    for i in freq_dict:
        a1.append(freq_dict[i])
        a1.sort()
    a1.reverse()
    for a in a1:
        for b in freq_dict:
            if freq_dict[b] == a:
                ans[b] = freq_dict[b]
    return ans

# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> Dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    >>> a = build_frequency_dict(bytes(1))
    >>> a == {0: 1}
    True
    """
    # TODO: Implement this function
    ans = {}
    for b in text:
        if b in ans:
            ans[b] += 1
        else:
            ans[b] = 1
    return ans


def build_huffman_tree(freq_dict: Dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    """
    if len(freq_dict) == 0:
        return HuffmanTree(None)
    elif len(freq_dict) == 1:
        dummy_tree = HuffmanTree((list(freq_dict.keys())[0] + 1) % 256)
        ans = HuffmanTree(None, HuffmanTree(list(freq_dict.keys())[0]),
                          dummy_tree)
        return ans
    else:
        new_freq = _sort_dict(freq_dict)
        a_name = 0
        tree_dict = {}
        while len(new_freq) > 1:
            two_mins = _get_two_mins(new_freq)
            two_mins_b = (HuffmanTree(two_mins[0]), HuffmanTree(two_mins[1]))
            if isinstance(two_mins[0], int) and isinstance(two_mins[1], int):
                ans = HuffmanTree(None, two_mins_b[0], two_mins_b[1])
                tree_dict['none' + str(a_name)] = ans
                none_frequency = new_freq[two_mins[0]] + new_freq[two_mins[1]]
                new_freq['none' + str(a_name)] = none_frequency
                del new_freq[two_mins[0]]
                del new_freq[two_mins[1]]
                new_freq = _sort_dict(new_freq)
            elif not isinstance(two_mins[0], int) and \
                    not isinstance(two_mins[1], int):
                ans = HuffmanTree(None, tree_dict[two_mins[0]],
                                  tree_dict[two_mins[1]])
                tree_dict['none' + str(a_name)] = ans
                none_frequency = new_freq[two_mins[0]] + \
                                 new_freq[two_mins[1]]
                new_freq['none' + str(a_name)] = none_frequency
                del new_freq[two_mins[0]]
                del new_freq[two_mins[1]]
                new_freq = _sort_dict(new_freq)
            elif not isinstance(two_mins[1], int):
                ans = HuffmanTree(None, two_mins_b[0],
                                  tree_dict[two_mins[1]])
                tree_dict['none' + str(a_name)] = ans
                none_frequency = new_freq[two_mins[0]] + \
                                 new_freq[two_mins[1]]
                new_freq['none' + str(a_name)] = none_frequency
                del new_freq[two_mins[0]]
                del new_freq[two_mins[1]]
                new_freq = _sort_dict(new_freq)
            else:
                ans = HuffmanTree(None, tree_dict[two_mins[0]],
                                  two_mins_b[1])
                tree_dict['none' + str(a_name)] = ans
                none_frequency = new_freq[two_mins[0]] + \
                                 new_freq[two_mins[1]]
                new_freq['none' + str(a_name)] = none_frequency
                del new_freq[two_mins[0]]
                del new_freq[two_mins[1]]
                new_freq = _sort_dict(new_freq)
            a_name += 1
        return ans


def get_codes(tree: HuffmanTree) -> Dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    >>> tree2 = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> a = get_codes(tree2)
    >>> a == {2: "0", 3: "10", 7: "11"}
    True
    """
    # TODO: Implement this function
    ans = _set_symbols(tree, '')
    return ans


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    # TODO: Implement this function
    lst = _set_node(tree)
    for t in range(len(lst)):
        lst[t].number = t


def avg_length(tree: HuffmanTree, freq_dict: Dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """
    # TODO: Implement this function
    dic1 = get_codes(tree)
    dic2 = {}
    for i in dic1:
        dic2[i] = len(dic1[i])
    num1 = 0
    for i in dic2:
        num1 += freq_dict[i] * dic2[i]
    num2 = 0
    for i in freq_dict:
        if i in dic2:
            num2 += freq_dict[i]
    return num1 / num2


def compress_bytes(text: bytes, codes: Dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    # TODO: Implement this function
    text_code = ''
    integers = []
    for i in text:
        text_code += codes[i]
        while len(text_code) > 8:
            integers.append(bits_to_byte(text_code[:8]))
            text_code = text_code[8:]
    integers.append(bits_to_byte(text_code))
    return bytes(integers)


def tree_to_bytes(tree: HuffmanTree) -> List[Union[int, Any]]:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    """
    # TODO: Implement this function
    lst = _set_node(tree)
    ans = []
    for i in lst:
        if i.left.is_leaf():
            ans.append(0)
            ans.append(i.left.symbol)
        else:
            ans.append(1)
            ans.append(i.left.number)
        if i.right.is_leaf():
            ans.append(0)
            ans.append(i.right.symbol)
        else:
            ans.append(1)
            ans.append(i.right.number)
    return bytes(ans)


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree) +
              int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: List[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    >>> lst2 = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 3), ReadNode(0, 6, 1, 0)]
    >>> generate_tree_general(lst2, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), HuffmanTree(None, HuffmanTree(6, None, None), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None))))
    """
    # TODO: Implement this function
    node = node_lst[root_index]
    if node.l_type == 1 and node.r_type == 1:
        root = HuffmanTree(None)
        root.left = generate_tree_general(node_lst, node.l_data)
        root.right = generate_tree_general(node_lst, node.r_data)
    elif node.l_type == 0 and node.r_type == 0:
        root = HuffmanTree(None, HuffmanTree(node.l_data),
                           HuffmanTree(node.r_data))
    elif node.l_type == 0:
        root = HuffmanTree(None, HuffmanTree(node.l_data), None)
        root.right = generate_tree_general(node_lst, node.r_data)
    else:
        root = HuffmanTree(None, None, HuffmanTree(node.r_data))
        root.left = generate_tree_general(node_lst, node.l_data)

    return root


def generate_tree_postorder(node_lst: List[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    >>> lst2 = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(0, 6, 1, 0), ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst2, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), HuffmanTree(None, HuffmanTree(6, None, None), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None))))
    """
    # TODO: Implement this function
    copy_list = node_lst.copy()
    root_index += 1
    ans = _generate_tree_postorder_helper(copy_list)

    return ans


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    """
    # TODO: Implement this function
    dic = get_codes(tree)
    tup = _switch_dict(dic)
    dic = tup[0]
    len_lst = tup[1]
    s = ''
    goal_size = size
    ans = []
    for byte in text:
        s += byte_to_bits(byte)
        t = _decompress_bytes_helper(s, dic, size, len_lst)
        ans.extend(t[0])
        size = goal_size - len(ans)
        s = t[1]
    return bytes(ans)


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: Dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    >>> right2 = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> left2 = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree2 = HuffmanTree(None, left2, right2)
    >>> freq2 = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree2, freq2)
    2.49
    >>> improve_tree(tree2, freq2)
    >>> avg_length(tree2, freq2)
    2.31
    """
    # TODO: Implement this function
    symbol_code = get_codes(tree)
    symbol_length = {}
    for i in symbol_code:
        symbol_length[i] = len(symbol_code[i])
    symbol_length = _sort_dict_reverse(symbol_length)
    lst = _list_leaves(tree)
    freq_dict = _sort_dict(freq_dict)
    lst1 = []
    for a in symbol_length:
        for b in lst:
            if a == b.symbol:
                lst1.append(b)
    lst2 = []
    for c in freq_dict:
        lst2.append(HuffmanTree(c))
    for d in range(len(lst1)):
        lst1[d].symbol = lst2[d].symbol


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input("Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print("Compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print("Decompressed {} in {} seconds."
              .format(fname, time.time() - start))
