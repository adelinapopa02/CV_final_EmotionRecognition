#ifndef COLORS_H
#define COLORS_H

#include <string>

// ANSI Color Codes - Header Only Implementation
namespace Colors {
    // Color constants
    inline const std::string RESET = "\033[0m";
    inline const std::string RED = "\033[31m";
    inline const std::string GREEN = "\033[32m";
    inline const std::string YELLOW = "\033[33m";
    inline const std::string BLUE = "\033[34m";
    inline const std::string MAGENTA = "\033[35m";
    inline const std::string CYAN = "\033[36m";
    inline const std::string WHITE = "\033[37m";
    inline const std::string BOLD = "\033[1m";
    
    // Helper functions - inline to avoid multiple definitions
    inline std::string success(const std::string& text) { 
        return GREEN + text + RESET; 
    }
    
    inline std::string error(const std::string& text) { 
        return RED + text + RESET; 
    }
    
    inline std::string warning(const std::string& text) { 
        return YELLOW + text + RESET; 
    }
    
    inline std::string info(const std::string& text) { 
        return CYAN + text + RESET; 
    }
    
    inline std::string bold(const std::string& text) { 
        return BOLD + text + RESET; 
    }
}

#endif // COLORS_H