//
// Created by wangzijian on 8/5/24.
//
// reference: https://github.com/leejet/stable-diffusion.cpp

#ifndef LITE_AI_TOOLKIT_TOKENIZER_H
#define LITE_AI_TOOLKIT_TOKENIZER_H


#include "iostream"
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <assert.h>
#include <inttypes.h>
#include <stdarg.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "codecvt"


namespace trt_tokenizer
{
    enum SDVersion {
        VERSION_1_x,
        VERSION_2_x,
        VERSION_XL,
        VERSION_SVD,
        VERSION_COUNT,
    };

    std::string rtrim(const std::string& s) {
        auto it = std::find_if(s.rbegin(), s.rend(), [](int ch) {
            return !std::isspace(ch);
        });
        return std::string(s.begin(), it.base());
    }

    std::string ltrim(const std::string& s) {
        auto it = std::find_if(s.begin(), s.end(), [](int ch) {
            return !std::isspace(ch);
        });
        return std::string(it, s.end());
    }

    std::string trim(const std::string& s) {
        return rtrim(ltrim(s));
    }

    bool ends_with(const std::string& str, const std::string& ending) {
        if (str.length() >= ending.length()) {
            return (str.compare(str.length() - ending.length(), ending.length(), ending) == 0);
        } else {
            return false;
        }
    }


    std::u32string unicode_value_to_utf32(int unicode_value) {
        std::u32string utf32_string = {static_cast<char32_t>(unicode_value)};
        return utf32_string;
    }


    std::u32string utf8_to_utf32(const std::string& utf8_str) {
        std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
        return converter.from_bytes(utf8_str);
    }

    std::string utf32_to_utf8(const std::u32string& utf32_str) {
        std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
        return converter.to_bytes(utf32_str);
    }


/*================================================== CLIPTokenizer ===================================================*/

    std::pair<std::unordered_map<std::string, float>, std::string> extract_and_remove_lora(std::string text) {
        std::regex re("<lora:([^:]+):([^>]+)>");
        std::smatch matches;
        std::unordered_map<std::string, float> filename2multiplier;

        while (std::regex_search(text, matches, re)) {
            std::string filename = matches[1].str();
            float multiplier     = std::stof(matches[2].str());

            text = std::regex_replace(text, re, "", std::regex_constants::format_first_only);

            if (multiplier == 0.f) {
                continue;
            }

            if (filename2multiplier.find(filename) == filename2multiplier.end()) {
                filename2multiplier[filename] = multiplier;
            } else {
                filename2multiplier[filename] += multiplier;
            }
        }

        return std::make_pair(filename2multiplier, text);
    }

    const std::string UNK_TOKEN = "<|endoftext|>";
    const std::string BOS_TOKEN = "<|startoftext|>";
    const std::string EOS_TOKEN = "<|endoftext|>";
    const std::string PAD_TOEKN = "<|endoftext|>";

    const int UNK_TOKEN_ID = 49407;
    const int BOS_TOKEN_ID = 49406;
    const int EOS_TOKEN_ID = 49407;
    const int PAD_TOKEN_ID = 49407;

    std::vector<std::pair<int, std::u32string>> bytes_to_unicode() {
        std::vector<std::pair<int, std::u32string>> byte_unicode_pairs;
        std::set<int> byte_set;
        for (int b = static_cast<int>('!'); b <= static_cast<int>('~'); ++b) {
            byte_set.insert(b);
            byte_unicode_pairs.push_back(std::pair<int, std::u32string>(b, unicode_value_to_utf32(b)));
        }
        for (int b = 161; b <= 172; ++b) {
            byte_set.insert(b);
            byte_unicode_pairs.push_back(std::pair<int, std::u32string>(b, unicode_value_to_utf32(b)));
        }
        for (int b = 174; b <= 255; ++b) {
            byte_set.insert(b);
            byte_unicode_pairs.push_back(std::pair<int, std::u32string>(b, unicode_value_to_utf32(b)));
        }
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (byte_set.find(b) == byte_set.end()) {
                byte_unicode_pairs.push_back(std::pair<int, std::u32string>(b, unicode_value_to_utf32(n + 256)));
                ++n;
            }
        }
        // LOG_DEBUG("byte_unicode_pairs %d", byte_unicode_pairs.size());
        return byte_unicode_pairs;
    }

// Ref: https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py

    typedef std::function<bool(std::string&, std::vector<int32_t>&)> on_new_token_cb_t;

    class CLIPTokenizer {
    private:
        SDVersion version = VERSION_1_x;
        std::map<int, std::u32string> byte_encoder;
        std::map<std::u32string, int> byte_decoder;
        std::map<std::u32string, int> encoder;
        std::map<int, std::u32string> decoder;
        std::map<std::pair<std::u32string, std::u32string>, int> bpe_ranks;
        std::regex pat;
        int encoder_len;
        int bpe_len;

        static std::string strip(const std::string& str) {
            std::string::size_type start = str.find_first_not_of(" \t\n\r\v\f");
            std::string::size_type end   = str.find_last_not_of(" \t\n\r\v\f");

            if (start == std::string::npos) {
                // String contains only whitespace characters
                return "";
            }

            return str.substr(start, end - start + 1);
        }

        static std::string whitespace_clean(std::string text) {
            text = std::regex_replace(text, std::regex(R"(\s+)"), " ");
            text = strip(text);
            return text;
        }

        static std::set<std::pair<std::u32string, std::u32string>> get_pairs(const std::vector<std::u32string>& subwords) {
            std::set<std::pair<std::u32string, std::u32string>> pairs;
            if (subwords.size() == 0) {
                return pairs;
            }
            std::u32string prev_subword = subwords[0];
            for (int i = 1; i < subwords.size(); i++) {
                std::u32string subword = subwords[i];
                std::pair<std::u32string, std::u32string> pair(prev_subword, subword);
                pairs.insert(pair);
                prev_subword = subword;
            }
            return pairs;
        }

    public:
        CLIPTokenizer(SDVersion version = VERSION_1_x)
                : version(version) {}

        void load_from_merges(const std::string& merges_utf8_str) {
            auto byte_unicode_pairs = bytes_to_unicode();
            // printf("byte_unicode_pairs have %lu pairs \n", byte_unicode_pairs.size());
            byte_encoder = std::map<int, std::u32string>(byte_unicode_pairs.begin(), byte_unicode_pairs.end());
            for (auto& pair : byte_unicode_pairs) {
                byte_decoder[pair.second] = pair.first;
            }
            // for (auto & pair: byte_unicode_pairs) {
            //     std::cout << pair.first << ": " << pair.second << std::endl;
            // }
            std::vector<std::u32string> merges;
            size_t start = 0;
            size_t pos;
            std::u32string merges_utf32_str = utf8_to_utf32(merges_utf8_str);
            while ((pos = merges_utf32_str.find('\n', start)) != std::string::npos) {
                merges.push_back(merges_utf32_str.substr(start, pos - start));
                start = pos + 1;
            }
            // LOG_DEBUG("merges size %llu", merges.size());
//        GGML_ASSERT(merges.size() == 48895);
            merges = std::vector<std::u32string>(merges.begin() + 1, merges.end());
            std::vector<std::pair<std::u32string, std::u32string>> merge_pairs;
            for (const auto& merge : merges) {
                size_t space_pos = merge.find(' ');
                merge_pairs.emplace_back(merge.substr(0, space_pos), merge.substr(space_pos + 1));
                // LOG_DEBUG("%s", utf32_to_utf8(merge.substr(space_pos + 1)).c_str());
                // printf("%s :: %s | %s \n", utf32_to_utf8(merge).c_str(), utf32_to_utf8(merge.substr(0, space_pos)).c_str(),
                //                     utf32_to_utf8(merge.substr(space_pos + 1)).c_str());
            }
            std::vector<std::u32string> vocab;
            for (const auto& pair : byte_unicode_pairs) {
                vocab.push_back(pair.second);
            }
            for (const auto& pair : byte_unicode_pairs) {
                vocab.push_back(pair.second + utf8_to_utf32("</w>"));
            }
            for (const auto& merge : merge_pairs) {
                vocab.push_back(merge.first + merge.second);
            }
            vocab.push_back(utf8_to_utf32("<|startoftext|>"));
            vocab.push_back(utf8_to_utf32("<|endoftext|>"));
//        LOG_DEBUG("vocab size: %llu", vocab.size());
            int i = 0;
            for (const auto& token : vocab) {
                encoder[token] = i;
                decoder[i]     = token;
                i++;
            }
            encoder_len = i;

            auto it = encoder.find(utf8_to_utf32("img</w>"));
            if (it != encoder.end()) {
                std::cout<<" trigger word img already in vocab"<<std::endl;
            } else {
                std::cout<<" trigger word img already in vocab"<<std::endl;
            }

            int rank = 0;
            for (const auto& merge : merge_pairs) {
                bpe_ranks[merge] = rank++;
            }
            bpe_len = rank;
        };

        void add_token(const std::string& text) {
            std::u32string token = utf8_to_utf32(text);
            auto it              = encoder.find(token);
            if (it != encoder.end()) {
                encoder[token]       = encoder_len;
                decoder[encoder_len] = token;
                encoder_len++;
            }
        }

        std::u32string bpe(const std::u32string& token) {
            std::vector<std::u32string> word;

            for (int i = 0; i < token.size() - 1; i++) {
                word.emplace_back(1, token[i]);
            }
            word.push_back(token.substr(token.size() - 1) + utf8_to_utf32("</w>"));

            std::set<std::pair<std::u32string, std::u32string>> pairs = get_pairs(word);

            if (pairs.empty()) {
                return token + utf8_to_utf32("</w>");
            }

            while (true) {
                auto min_pair_iter = std::min_element(pairs.begin(),
                                                      pairs.end(),
                                                      [&](const std::pair<std::u32string, std::u32string>& a,
                                                          const std::pair<std::u32string, std::u32string>& b) {
                                                          if (bpe_ranks.find(a) == bpe_ranks.end()) {
                                                              return false;
                                                          } else if (bpe_ranks.find(b) == bpe_ranks.end()) {
                                                              return true;
                                                          }
                                                          return bpe_ranks.at(a) < bpe_ranks.at(b);
                                                      });

                const std::pair<std::u32string, std::u32string>& bigram = *min_pair_iter;

                if (bpe_ranks.find(bigram) == bpe_ranks.end()) {
                    break;
                }

                std::u32string first  = bigram.first;
                std::u32string second = bigram.second;
                std::vector<std::u32string> new_word;
                int32_t i = 0;

                while (i < word.size()) {
                    auto it = std::find(word.begin() + i, word.end(), first);
                    if (it == word.end()) {
                        new_word.insert(new_word.end(), word.begin() + i, word.end());
                        break;
                    }
                    new_word.insert(new_word.end(), word.begin() + i, it);
                    i = static_cast<int32_t>(std::distance(word.begin(), it));

                    if (word[i] == first && i < static_cast<int32_t>(word.size()) - 1 && word[i + 1] == second) {
                        new_word.push_back(first + second);
                        i += 2;
                    } else {
                        new_word.push_back(word[i]);
                        i += 1;
                    }
                }

                word = new_word;

                if (word.size() == 1) {
                    break;
                }
                pairs = get_pairs(word);
            }

            std::u32string result;
            for (int i = 0; i < word.size(); i++) {
                result += word[i];
                if (i != word.size() - 1) {
                    result += utf8_to_utf32(" ");
                }
            }

            return result;
        }

        std::vector<int> tokenize(std::string text,
                                  on_new_token_cb_t on_new_token_cb,
                                  size_t max_length = 0,
                                  bool padding      = false) {
            std::vector<int32_t> tokens = encode(text, on_new_token_cb);

            tokens.insert(tokens.begin(), BOS_TOKEN_ID);
            if (max_length > 0) {
                if (tokens.size() > max_length - 1) {
                    tokens.resize(max_length - 1);
                    tokens.push_back(EOS_TOKEN_ID);
                } else {
                    tokens.push_back(EOS_TOKEN_ID);
                    if (padding) {
                        int pad_token_id = PAD_TOKEN_ID;
                        if (version == VERSION_2_x) {
                            pad_token_id = 0;
                        }
                        tokens.insert(tokens.end(), max_length - tokens.size(), pad_token_id);
                    }
                }
            }

            return tokens;
        }

        std::string decode(const std::vector<int>& tokens) {
            std::string text = "";
            for (int t : tokens) {
                if (t == 49406 || t == 49407)
                    continue;
                std::u32string ts = decoder[t];
                // printf("%d, %s \n", t,  utf32_to_utf8(ts).c_str());
                std::string s = utf32_to_utf8(ts);
                if (s.length() >= 4 && ends_with(s, "</w>")) {
                    text += " " + s.replace(s.length() - 4, s.length() - 1, "");
                } else {
                    text += " " + s;
                }
            }
            // std::vector<unsigned char> bytes;
            // for (auto c : text){
            //     bytes.push_back(byte_decoder[c]);
            // }

            // std::string s((char *)bytes.data());
            // std::string s = "";
            return trim(text);
        }

        std::vector<int> encode(std::string text, on_new_token_cb_t on_new_token_cb) {
            std::string original_text = text;
            std::vector<int32_t> bpe_tokens;
            text = whitespace_clean(text);
            std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });

            std::regex pat(R"(<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[[:alpha:]]+|[[:digit:]]|[^[:space:][:alpha:][:digit:]]+)",
                           std::regex::icase);

            std::smatch matches;
            std::string str = text;
            std::vector<std::string> token_strs;
            while (std::regex_search(str, matches, pat)) {
                bool skip = on_new_token_cb(str, bpe_tokens);
                if (skip) {
                    continue;
                }
                for (auto& token : matches) {
                    std::string token_str = token.str();
                    std::u32string utf32_token;
                    for (int i = 0; i < token_str.length(); i++) {
                        char b = token_str[i];
                        utf32_token += byte_encoder[b];
                    }
                    auto bpe_strs = bpe(utf32_token);
                    size_t start  = 0;
                    size_t pos;
                    while ((pos = bpe_strs.find(' ', start)) != std::u32string::npos) {
                        auto bpe_str = bpe_strs.substr(start, pos - start);
                        bpe_tokens.push_back(encoder[bpe_str]);
                        token_strs.push_back(utf32_to_utf8(bpe_str));

                        start = pos + 1;
                    }
                    auto bpe_str = bpe_strs.substr(start, bpe_strs.size() - start);
                    bpe_tokens.push_back(encoder[bpe_str]);
                    token_strs.push_back(utf32_to_utf8(bpe_str));
                }
                str = matches.suffix();
            }
            std::stringstream ss;
            ss << "[";
            for (auto token : token_strs) {
                ss << "\"" << token << "\", ";
            }
            ss << "]";
            // LOG_DEBUG("split prompt \"%s\" to tokens %s", original_text.c_str(), ss.str().c_str());
            // printf("split prompt \"%s\" to tokens %s \n", original_text.c_str(), ss.str().c_str());
            return bpe_tokens;
        }
    };


    std::vector<std::pair<std::string, float>> parse_prompt_attention(const std::string& text) {
        std::vector<std::pair<std::string, float>> res;
        std::vector<int> round_brackets;
        std::vector<int> square_brackets;

        float round_bracket_multiplier  = 1.1f;
        float square_bracket_multiplier = 1 / 1.1f;

        std::regex re_attention(R"(\\\(|\\\)|\\\[|\\\]|\\\\|\\|\(|\[|:([+-]?[.\d]+)\)|\)|\]|[^\\()\[\]:]+|:)");
        std::regex re_break(R"(\s*\bBREAK\b\s*)");

        auto multiply_range = [&](int start_position, float multiplier) {
            for (int p = start_position; p < res.size(); ++p) {
                res[p].second *= multiplier;
            }
        };

        std::smatch m;
        std::string remaining_text = text;

        while (std::regex_search(remaining_text, m, re_attention)) {
            std::string text   = m[0];
            std::string weight = m[1];

            if (text == "(") {
                round_brackets.push_back((int)res.size());
            } else if (text == "[") {
                square_brackets.push_back((int)res.size());
            } else if (!weight.empty()) {
                if (!round_brackets.empty()) {
                    multiply_range(round_brackets.back(), std::stof(weight));
                    round_brackets.pop_back();
                }
            } else if (text == ")" && !round_brackets.empty()) {
                multiply_range(round_brackets.back(), round_bracket_multiplier);
                round_brackets.pop_back();
            } else if (text == "]" && !square_brackets.empty()) {
                multiply_range(square_brackets.back(), square_bracket_multiplier);
                square_brackets.pop_back();
            } else if (text == "\\(") {
                res.push_back({text.substr(1), 1.0f});
            } else {
                res.push_back({text, 1.0f});
            }

            remaining_text = m.suffix();
        }

        for (int pos : round_brackets) {
            multiply_range(pos, round_bracket_multiplier);
        }

        for (int pos : square_brackets) {
            multiply_range(pos, square_bracket_multiplier);
        }

        if (res.empty()) {
            res.push_back({"", 1.0f});
        }

        int i = 0;
        while (i + 1 < res.size()) {
            if (res[i].second == res[i + 1].second) {
                res[i].first += res[i + 1].first;
                res.erase(res.begin() + i + 1);
            } else {
                ++i;
            }
        }

        return res;
    }


#endif //LITE_AI_TOOLKIT_TOKENIZER_H

}
